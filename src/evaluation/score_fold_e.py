"""Score one trained fold of a step-E variant on its own test fold.

Inference as fixed by D1 (results/D1.md): flip TTA (original + 3 single-axis flips),
count-map averaging, 50% overlap -> "I2"; the no-TTA original-orientation map "I0" is
kept too. Both are saved as float16 under <run-dir>/oof_probs/{I0,I2}/. Per-case CSV:
  dice_d1    I2, threshold 0.5, keep components >= 10% of the largest  (frozen config)
  dice_e0cfg I0, threshold 0.5, largest connected component            (E0 config, descriptive)

    python src/evaluation/score_fold_e.py --fold 0 --run-dir outputs/cv_e1_4ch_2026-11-17 --recipe E1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_3d import UNet3D
from training.train_cv import RECIPES, FOLDS_CSV
from evaluation.infer_variants_cv import sw_pass, gaussian_map
from evaluation.d1_crossfit import keep_mask
from evaluation.evaluate_3d import git_commit, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MSK = PROJECT_ROOT / "data" / "processed_3d" / "masks"
EPS = 1e-6


def dice_post(prob, gt, thr, post):
    lab, n = ndi.label(prob >= thr)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    inter = np.bincount(lab[gt], minlength=n + 1)[1:]
    k = keep_mask(sizes, post)
    return float((2.0 * inter[k].sum() + EPS) / (sizes[k].sum() + gt.sum() + EPS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    args = p.parse_args()
    recipe = RECIPES[args.recipe]
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    fold_dir = run_dir / f"fold{args.fold}"
    out_csv = fold_dir / "test_per_case.csv"
    if out_csv.exists():
        print(f"{out_csv} exists; fold {args.fold} already scored.")
        return
    ckpt = fold_dir / "final_model.pt"
    folds = pd.read_csv(FOLDS_CSV)
    test = sorted(folds.loc[folds.fold == args.fold, "subject_id"])
    assert not set((fold_dir / "train_subjects.txt").read_text().split()) & set(test)
    for v in ("I0", "I2"):
        (run_dir / "oof_probs" / v).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    model = UNet3D(in_channels=recipe["in_channels"], out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    gmap = gaussian_map(device)
    img_dir = PROJECT_ROOT / recipe["image_dir"]

    rows = []
    for i, sid in enumerate(test, 1):
        img = torch.as_tensor(np.load(img_dir / f"{sid}.npy"), dtype=torch.float32, device=device)
        gt = np.load(MSK / f"{sid}.npy").astype(bool)
        acc = None
        for flip in (None, 1, 2, 3):
            x = img if flip is None else torch.flip(img, dims=(flip,))
            c, _ = sw_pass(model, x, 0.5, gmap)
            if flip is None:
                i0 = c.cpu().numpy()
            else:
                c = torch.flip(c, dims=(flip - 1,))
            acc = c if acc is None else acc + c
        i2 = (acc / 4).cpu().numpy()
        np.save(run_dir / "oof_probs" / "I0" / f"{sid}.npy", i0.astype(np.float16))
        np.save(run_dir / "oof_probs" / "I2" / f"{sid}.npy", i2.astype(np.float16))
        r = {"subject_id": sid, "fold": args.fold, "tumor_voxels": int(gt.sum()),
             "dice_d1": dice_post(i2, gt, 0.5, "rel10"),
             "dice_e0cfg": dice_post(i0, gt, 0.5, "lcc")}
        rows.append(r)
        print(f"score fold {args.fold}: {i}/{len(test)} {sid} dice_d1 {r['dice_d1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    (fold_dir / "eval_config.json").write_text(json.dumps({
        "fold": args.fold, "recipe": args.recipe, "checkpoint_sha256": sha256(ckpt),
        "inference": "I2: flip TTA (orig + 3 single-axis flips), count averaging, 50% overlap, AMP, batch-6 patches",
        "frozen_config": "I2, threshold 0.5, keep components >= 10% of largest (D1)",
        "git_commit": git_commit(), "date": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"fold {args.fold}: n={len(df)} mean dice_d1 {df.dice_d1.mean():.4f}  "
          f"mean dice_e0cfg {df.dice_e0cfg.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
