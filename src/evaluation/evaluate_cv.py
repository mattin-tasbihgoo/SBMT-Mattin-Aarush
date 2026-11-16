"""Score one trained CV fold on its own test fold (results/PROTOCOL.md).

Runs the frozen baseline inference (sliding window 128^3, 50% overlap,
count-map averaging, AMP, no TTA) with the fold's final checkpoint on the
fold's test subjects only. Saves float16 probability maps (for later
inference-only decisions made by cross-fitting, see PROTOCOL.md section 5) and a
per-case CSV with Dice at threshold 0.5 with and without largest-CC.

    python src/evaluation/evaluate_cv.py --fold 0 --run-dir outputs/cv_e0_v1recipe_2026-11-15
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_3d import UNet3D
from evaluation.sliding_window import sliding_window_inference
from evaluation.evaluate_3d import dice_3d, largest_connected_component, git_commit, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
THR = 0.5
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    fold_dir = run_dir / f"fold{args.fold}"
    out_csv = fold_dir / "test_per_case.csv"
    if out_csv.exists():
        print(f"{out_csv} exists; fold {args.fold} already scored.")
        return
    ckpt = fold_dir / "final_model.pt"
    assert ckpt.exists(), f"missing {ckpt}"

    folds = pd.read_csv(FOLDS_CSV)
    test = sorted(folds.loc[folds.fold == args.fold, "subject_id"])
    trained_on = set((fold_dir / "train_subjects.txt").read_text().split())
    assert not trained_on & set(test), "test subject found in this fold's training list"

    prob_dir = run_dir / "oof_probs"
    prob_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    model = UNet3D(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    rows = []
    for i, sid in enumerate(test, 1):
        image = np.load(DATA / "images" / f"{sid}.npy")
        gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
        prob = sliding_window_inference(model, image, patch_size=(128, 128, 128), overlap=0.5,
                                        device=device, amp_enabled=True)
        np.save(prob_dir / f"{sid}.npy", prob.astype(np.float16))
        raw = prob >= THR
        cc = largest_connected_component(raw)
        r = {"subject_id": sid, "fold": args.fold, "tumor_voxels": int(gt.sum()),
             "dice_cc": dice_3d(cc, gt), "dice_nocc": dice_3d(raw, gt),
             "pred_voxels": int(cc.sum()), "pred_voxels_nocc": int(raw.sum())}
        for t in SWEEP:
            r[f"dice_nocc_t{t:.2f}"] = dice_3d(prob >= t, gt)
        rows.append(r)
        print(f"eval fold {args.fold}: {i}/{len(test)} {sid} dice_cc {r['dice_cc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    (fold_dir / "eval_config.json").write_text(json.dumps({
        "fold": args.fold, "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(ckpt),
        "inference": "sliding window 128^3, overlap 0.5, count-map averaging, AMP, no TTA",
        "threshold": THR, "post": "largest connected component (dice_cc); none (dice_nocc)",
        "git_commit": git_commit(), "date": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"fold {args.fold}: n={len(df)} mean dice_cc {df.dice_cc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
