"""Evaluate a trained 3D U-Net on the 96 frozen validation subjects.

Usage (from the repo root):
  python src/evaluation/evaluate_3d.py --model v1       --out-dir outputs/audit_2026-11-15/v1
  python src/evaluation/evaluate_3d.py --model v2       --out-dir outputs/audit_2026-11-15/v2
  python src/evaluation/evaluate_3d.py --model ensemble --out-dir outputs/audit_2026-11-15/ensemble
  python src/evaluation/evaluate_3d.py --model v2 --checkpoint outputs/3d_unet_v2/checkpoints/latest_model_3d_unet_v2.pt \
      --out-dir outputs/audit_2026-11-15/v2_latest

Defaults reproduce the April 2026 configuration: sliding window 128^3, 50 %
overlap, plain (count-map) averaging, AMP on CUDA, threshold 0.5, keep the
largest connected component, no TTA. Ensemble = mean of V1 and V2 probability
maps.

The script refuses to write into a directory that already holds results, so
the April CSVs under outputs/3d_unet_*/evaluation/ cannot be overwritten.

Outputs in --out-dir:
  subject_metrics_3d.csv   per case: dice_3d (thr 0.5 + largest CC, the April
                           metric) plus no-CC Dice, per-threshold Dice, voxel
                           counts and number of predicted components
  threshold_sweep_3d.csv   same format as the April files
  run_config.json          checkpoints (with sha256), settings, versions
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as cc_label
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_3d import UNet3D
from models.model_3d_v2 import UNet3D_V2
from evaluation.sliding_window import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
META_CSV = DATA_ROOT / "metadata.csv"

REFERENCE_SPLIT_RUN_NAME = "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"
REFERENCE_SPLIT_LOG_DIR = PROJECT_ROOT / "outputs" / REFERENCE_SPLIT_RUN_NAME / "logs"
REFERENCE_VAL_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "val_subjects.txt"

CHECKPOINTS = {
    "v1": PROJECT_ROOT / "outputs" / "3d_unet_v1" / "checkpoints" / "best_model_3d_unet_v1.pt",
    "v2": PROJECT_ROOT / "outputs" / "3d_unet_v2" / "checkpoints" / "best_model_3d_unet_v2.pt",
}

# ── Inference settings (April 2026 configuration) ─────────────────────
PATCH_SIZE = (128, 128, 128)
OVERLAP = 0.5
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
BASE_THRESHOLD = 0.50
RANDOM_SEED = 42
USE_TTA = False
USE_CC_POSTPROC = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_list(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def resolve_val_subjects(all_subjects: list[str]) -> list[str]:
    """The frozen 96-subject list. No random fallback: a missing file is an error."""
    if not REFERENCE_VAL_SUBJECTS_TXT.exists():
        raise FileNotFoundError(f"Frozen split file missing: {REFERENCE_VAL_SUBJECTS_TXT}")
    val = load_subject_list(REFERENCE_VAL_SUBJECTS_TXT)
    missing = sorted(set(val) - set(all_subjects))
    if missing:
        raise RuntimeError("Reference val subjects not in metadata:\n" + "\n".join(missing[:20]))
    return val


def dice_3d(pred_bin: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
    pred_bin = pred_bin.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred_bin, target).sum()
    denom = pred_bin.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def predict_with_tta(model, image, patch_size, overlap, device, amp_enabled):
    """Average predictions over original + 3 single-axis flips."""
    prob = sliding_window_inference(model, image, patch_size, overlap, device, amp_enabled)
    for axis_img, axis_prob in [(1, 0), (2, 1), (3, 2)]:
        flipped_img = np.flip(image, axis=axis_img).copy()
        flipped_prob = sliding_window_inference(
            model, flipped_img, patch_size, overlap, device, amp_enabled
        )
        prob = prob + np.flip(flipped_prob, axis=axis_prob).copy()
    return prob / 4.0


def connected_components(binary_mask):
    labeled, n = cc_label(binary_mask)
    return labeled, n


def largest_connected_component(binary_mask):
    """Keep only the largest connected component in a binary volume."""
    labeled, n = cc_label(binary_mask)
    if n <= 1:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


def build_model(version: str, device: str):
    if version == "v2":
        return UNet3D_V2(in_channels=2, out_channels=1, deep_supervision=True).to(device)
    if version == "v1":
        return UNet3D(in_channels=2, out_channels=1).to(device)
    raise ValueError(version)


def infer_subject(model, image, patch_size, overlap, device, amp_enabled):
    if USE_TTA:
        return predict_with_tta(model, image, patch_size, overlap, device, amp_enabled)
    return sliding_window_inference(model, image, patch_size, overlap, device, amp_enabled)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["v1", "v2", "ensemble"])
    p.add_argument("--out-dir", required=True, type=Path,
                   help="New directory for results; must not already contain results.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override the checkpoint for --model v1 or v2 (e.g. a latest_ checkpoint).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is not None and args.model == "ensemble":
        raise SystemExit("--checkpoint applies to a single model, not the ensemble")

    eval_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir)
    if (eval_dir / "subject_metrics_3d.csv").exists():
        raise SystemExit(f"Refusing to overwrite existing results in {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)

    df = pd.read_csv(META_CSV)
    all_subjects = sorted(df["subject_id"].unique().tolist())
    val_subjects = resolve_val_subjects(all_subjects)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = device == "cuda"

    if args.model == "ensemble":
        members = [("v1", CHECKPOINTS["v1"]), ("v2", CHECKPOINTS["v2"])]
    else:
        members = [(args.model, args.checkpoint or CHECKPOINTS[args.model])]
    members = [(v, Path(cp) if Path(cp).is_absolute() else PROJECT_ROOT / cp) for v, cp in members]

    models = []
    for version, cp in members:
        m = build_model(version, device)
        m.load_state_dict(torch.load(cp, map_location=device, weights_only=True), strict=True)
        m.eval()
        models.append(m)

    print(f"Model: {args.model}  |  members: {[str(cp.relative_to(PROJECT_ROOT)) for _, cp in members]}")
    print(f"Val subjects: {len(val_subjects)}  |  split: {REFERENCE_VAL_SUBJECTS_TXT}")
    print(f"Device: {device}  |  TTA: {USE_TTA}  |  largest-CC: {USE_CC_POSTPROC}")
    print(f"Out: {eval_dir}")

    config = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": args.model,
        "checkpoints": [
            {"version": v, "path": str(cp.relative_to(PROJECT_ROOT)), "sha256": sha256(cp)}
            for v, cp in members
        ],
        "val_subjects_file": str(REFERENCE_VAL_SUBJECTS_TXT.relative_to(PROJECT_ROOT)),
        "n_subjects": len(val_subjects),
        "patch_size": PATCH_SIZE, "overlap": OVERLAP, "blend": "uniform count-map average",
        "thresholds": THRESHOLDS, "base_threshold": BASE_THRESHOLD,
        "tta": USE_TTA, "largest_cc": USE_CC_POSTPROC, "amp": amp_enabled,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "numpy": np.__version__},
    }
    (eval_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    thresholds = sorted(set(float(t) for t in THRESHOLDS + [BASE_THRESHOLD]))
    threshold_results = {thr: [] for thr in thresholds}
    subject_rows = []

    for sid in tqdm(val_subjects, desc=f"Evaluating 3D [{args.model}]"):
        image = np.load(str(IMG_DIR / f"{sid}.npy"))
        mask = np.load(str(MSK_DIR / f"{sid}.npy"))

        prob_acc = np.zeros_like(mask, dtype=np.float32)
        for m in models:
            prob_acc += infer_subject(m, image, PATCH_SIZE, OVERLAP, device, amp_enabled)
        pred_prob = prob_acc / len(models)

        row = {"subject_id": sid, "tumor_voxels": int(mask.sum())}
        for thr in thresholds:
            raw_bin = (pred_prob >= thr).astype(np.uint8)
            pred_bin = largest_connected_component(raw_bin) if USE_CC_POSTPROC else raw_bin
            d = dice_3d(pred_bin, mask)
            threshold_results[thr].append(d)
            row[f"dice_t{thr:.2f}"] = d
            if abs(thr - BASE_THRESHOLD) < 1e-12:
                row["dice_3d"] = d
                row["dice_nocc"] = dice_3d(raw_bin, mask)
                row["pred_voxels"] = int(pred_bin.sum())
                row["pred_voxels_nocc"] = int(raw_bin.sum())
                row["n_components"] = int(connected_components(raw_bin)[1])
        subject_rows.append(row)

    cols = ["subject_id", "tumor_voxels", "dice_3d", "dice_nocc", "pred_voxels",
            "pred_voxels_nocc", "n_components"] + [f"dice_t{t:.2f}" for t in thresholds]
    subject_df = pd.DataFrame(subject_rows)[cols]
    # Keep evaluation order (= frozen val_subjects.txt order) in a column, then sort like April.
    subject_df.insert(1, "val_order", range(len(subject_df)))
    subject_df = subject_df.sort_values("subject_id").reset_index(drop=True)
    subject_df.to_csv(eval_dir / "subject_metrics_3d.csv", index=False)

    thr_rows = []
    for thr in thresholds:
        arr = np.array(threshold_results[thr])
        thr_rows.append({
            "threshold": thr,
            "mean_dice": float(arr.mean()),
            "median_dice": float(np.median(arr)),
            "std_dice": float(arr.std()),
        })
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(eval_dir / "threshold_sweep_3d.csv", index=False)

    base_arr = np.array(threshold_results[BASE_THRESHOLD])
    print(f"\n3D Dice @ threshold {BASE_THRESHOLD} (largest CC={USE_CC_POSTPROC}):")
    print(f"  Mean:   {base_arr.mean():.4f} +/- {base_arr.std():.4f}")
    print(f"  Median: {np.median(base_arr):.4f}")
    print(f"\nThreshold sweep:")
    print(thr_df.to_string(index=False))
    print(f"\nSaved to: {eval_dir}")


if __name__ == "__main__":
    main()
