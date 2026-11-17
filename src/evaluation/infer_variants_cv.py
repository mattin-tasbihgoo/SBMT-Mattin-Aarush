"""D1: inference variants for every subject from its own E0 fold model (results/RUNS.md, D1 pre-registration).

Per subject (only with the model of the fold it was held out from):
  pass A, 50% overlap, original + 3 single-axis flips, accumulating both a count map
  and a Gaussian importance map (sigma = patch/8, nnU-Net style) from the same patch outputs:
    I0 count, no TTA (must equal E0's stored maps; checked, not re-saved)
    I1 Gaussian, no TTA
    I2 count, flip TTA (mean of the 4 orientations, as evaluate_3d.py --tta)
    I3 Gaussian, flip TTA
  pass B, 75% overlap, original orientation:
    I4 count
    I5 Gaussian
Saves float16 probability maps to outputs/<tag>/probs/I<n>/<sid>.npy. Resumable.

    python src/evaluation/infer_variants_cv.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_3d import UNet3D
from evaluation.evaluate_3d import git_commit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
E0 = PROJECT_ROOT / "outputs" / "cv_e0_v1recipe_2026-11-15"
OUT = PROJECT_ROOT / "outputs" / "d1_inference_2026-11-16"
PATCH = 128
BATCH = 6  # patches per forward pass (InstanceNorm: per-sample, so batching changes no math)
SAVED = ("I1", "I2", "I3", "I4", "I5")


def starts(length, patch, step):  # identical to sliding_window.py
    s = list(range(0, max(length - patch + 1, 1), step))
    if s[-1] + patch < length:
        s.append(length - patch)
    return s


def gaussian_map(device):
    x = torch.arange(PATCH, dtype=torch.float32, device=device)
    g = torch.exp(-((x - PATCH // 2) ** 2) / (2 * (PATCH / 8) ** 2))
    g3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return g3 / g3.max()


@torch.no_grad()
def sw_pass(model, img, overlap, gmap):
    """img: (C,H,W,D) cuda float32. Returns (count-averaged, Gaussian-averaged) probs, float32 cuda."""
    _, H, W, D = img.shape
    step = max(1, int(PATCH * (1 - overlap)))
    psum = torch.zeros((H, W, D), device=img.device)
    cnt = torch.zeros_like(psum)
    gsum = torch.zeros_like(psum)
    gw = torch.zeros_like(psum)
    coords = [(h0, w0, d0) for h0 in starts(H, PATCH, step)
              for w0 in starts(W, PATCH, step) for d0 in starts(D, PATCH, step)]
    for b in range(0, len(coords), BATCH):
        chunk = coords[b:b + BATCH]
        x = torch.stack([img[:, h0:h0 + PATCH, w0:w0 + PATCH, d0:d0 + PATCH] for h0, w0, d0 in chunk])
        with autocast(device_type="cuda"):
            logits = model(x)
        probs = torch.sigmoid(logits)[:, 0].float()  # same order of ops as sliding_window.py
        for (h0, w0, d0), prob in zip(chunk, probs):
            sl = (slice(h0, h0 + PATCH), slice(w0, w0 + PATCH), slice(d0, d0 + PATCH))
            psum[sl] += prob
            cnt[sl] += 1.0
            gsum[sl] += prob * gmap
            gw[sl] += gmap
    return psum / torch.clamp(cnt, min=1e-8), gsum / torch.clamp(gw, min=1e-8)


def main():
    folds = pd.read_csv(FOLDS_CSV)
    for v in SAVED:
        (OUT / "probs" / v).mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    gmap = gaussian_map(device)
    i0_maxdiff = []
    for k in range(5):
        test = sorted(folds.loc[folds.fold == k, "subject_id"])
        todo = [s for s in test if not all((OUT / "probs" / v / f"{s}.npy").exists() for v in SAVED)]
        if not todo:
            continue
        model = UNet3D(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(torch.load(E0 / f"fold{k}" / "final_model.pt", map_location=device, weights_only=True))
        model.eval()
        for i, sid in enumerate(todo, 1):
            img = torch.as_tensor(np.load(DATA / "images" / f"{sid}.npy"), dtype=torch.float32, device=device)
            c_sum = torch.zeros(img.shape[1:], device=device)
            g_sum = torch.zeros_like(c_sum)
            for flip in (None, 1, 2, 3):
                x = img if flip is None else torch.flip(img, dims=(flip,))
                c, g = sw_pass(model, x, 0.5, gmap)
                if flip is not None:
                    c, g = torch.flip(c, dims=(flip - 1,)), torch.flip(g, dims=(flip - 1,))
                else:
                    i0 = c.cpu().numpy().astype(np.float16)
                    e0 = np.load(E0 / "oof_probs" / f"{sid}.npy")
                    i0_maxdiff.append(float(np.abs(i0.astype(np.float32) - e0.astype(np.float32)).max()))
                    np.save(OUT / "probs" / "I1" / f"{sid}.npy", g.cpu().numpy().astype(np.float16))
                c_sum += c
                g_sum += g
            np.save(OUT / "probs" / "I2" / f"{sid}.npy", (c_sum / 4).cpu().numpy().astype(np.float16))
            np.save(OUT / "probs" / "I3" / f"{sid}.npy", (g_sum / 4).cpu().numpy().astype(np.float16))
            c, g = sw_pass(model, img, 0.75, gmap)
            np.save(OUT / "probs" / "I4" / f"{sid}.npy", c.cpu().numpy().astype(np.float16))
            np.save(OUT / "probs" / "I5" / f"{sid}.npy", g.cpu().numpy().astype(np.float16))
            print(f"infer fold {k}: {i}/{len(todo)} {sid} | I0 vs E0 max|diff| {i0_maxdiff[-1]:.1e}", flush=True)
    if i0_maxdiff:
        print(f"I0 reproduction vs E0 maps: max |diff| over {len(i0_maxdiff)} subjects = {max(i0_maxdiff):.2e}")
    (OUT / "infer_config.json").write_text(json.dumps({
        "variants": {"I0": "E0 stored maps (count, 0.5 overlap, no TTA)", "I1": "Gaussian, 0.5",
                     "I2": "count, 0.5, flip TTA", "I3": "Gaussian, 0.5, flip TTA",
                     "I4": "count, 0.75", "I5": "Gaussian, 0.75"},
        "gaussian": "sigma = patch/8, centre patch//2, normalised to max 1",
        "models": "E0 fold final_model.pt, each subject from its own test fold",
        "i0_vs_e0_max_abs_diff": max(i0_maxdiff) if i0_maxdiff else None,
        "git_commit": git_commit(), "date": datetime.now().isoformat(timespec="seconds")}, indent=2))


if __name__ == "__main__":
    main()
