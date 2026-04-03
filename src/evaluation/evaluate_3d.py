"""Evaluate a trained 3D U-Net on validation subjects.

Supports:
  - Test-time augmentation (TTA): flip along 3 axes, average 4 probability maps
  - Connected component post-processing: keep only the largest predicted region
  - Checkpoint ensemble: average predictions from multiple checkpoints
  - Model version selection: v1 (original) or v2 (larger + deep supervision)
"""

import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as cc_label
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_3d import UNet3D
from evaluation.sliding_window import sliding_window_inference

try:
    from models.model_3d_v2 import UNet3D_V2
except ImportError:
    UNet3D_V2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Experiment ────────────────────────────────────────────────────────
EXPERIMENT_NAME = "3d_unet_ensemble_v1v2"
MODEL_VERSION = "ensemble"          # "v1", "v2", or "ensemble"

DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
META_CSV = DATA_ROOT / "metadata.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"
EVAL_DIR = OUT_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_SPLIT_RUN_NAME = "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"
REFERENCE_SPLIT_LOG_DIR = PROJECT_ROOT / "outputs" / REFERENCE_SPLIT_RUN_NAME / "logs"
REFERENCE_VAL_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "val_subjects.txt"

# ── Inference settings ────────────────────────────────────────────────
PATCH_SIZE = (128, 128, 128)
OVERLAP = 0.5
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
BASE_THRESHOLD = 0.50
RANDOM_SEED = 42
VAL_FRAC = 0.20

# ── Phase 1 enhancements ─────────────────────────────────────────────
USE_TTA = False             # TTA doubles inference time; ensemble already covers variance
USE_CC_POSTPROC = True      # keep only largest connected component
ENSEMBLE_CHECKPOINTS = [
    PROJECT_ROOT / "outputs" / "3d_unet_v1" / "checkpoints" / "best_model_3d_unet_v1.pt",
    PROJECT_ROOT / "outputs" / "3d_unet_v2" / "checkpoints" / "best_model_3d_unet_v2.pt",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_list(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def resolve_val_subjects(all_subjects: list[str]):
    if REFERENCE_VAL_SUBJECTS_TXT.exists():
        val = load_subject_list(REFERENCE_VAL_SUBJECTS_TXT)
        missing = sorted(set(val) - set(all_subjects))
        if missing:
            raise RuntimeError(
                "Reference val subjects not in metadata:\n" + "\n".join(missing[:20])
            )
        return val, f"loaded from {REFERENCE_SPLIT_LOG_DIR}"

    subjects = sorted(all_subjects)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * VAL_FRAC))
    return subjects[:n_val], "deterministic RANDOM_SEED/VAL_FRAC fallback"


def dice_3d(pred_bin: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
    pred_bin = pred_bin.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred_bin, target).sum()
    denom = pred_bin.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def resolve_checkpoint() -> Path:
    best = CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt"
    latest = CKPT_DIR / f"latest_model_{EXPERIMENT_NAME}.pt"
    if best.exists():
        return best
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")


# ── Phase 1 helpers ──────────────────────────────────────────────────

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


def largest_connected_component(binary_mask):
    """Keep only the largest connected component in a binary volume."""
    labeled, n = cc_label(binary_mask)
    if n <= 1:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


def build_model(device, checkpoint_path=None):
    """Build a model appropriate for the given checkpoint path, or MODEL_VERSION."""
    cp = Path(checkpoint_path) if checkpoint_path else None
    # Detect V2 by checkpoint filename or explicit MODEL_VERSION
    is_v2 = (MODEL_VERSION == "v2") or (
        cp is not None and "v2" in cp.name
    )
    if is_v2:
        if UNet3D_V2 is None:
            raise ImportError("model_3d_v2.py not found — create it first.")
        return UNet3D_V2(in_channels=2, out_channels=1, deep_supervision=True).to(device)
    return UNet3D(in_channels=2, out_channels=1).to(device)


def infer_subject(model, image, patch_size, overlap, device, amp_enabled):
    """Run inference on a single subject, with or without TTA."""
    if USE_TTA:
        return predict_with_tta(model, image, patch_size, overlap, device, amp_enabled)
    return sliding_window_inference(model, image, patch_size, overlap, device, amp_enabled)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(META_CSV)
    all_subjects = sorted(df["subject_id"].unique().tolist())
    val_subjects, split_source = resolve_val_subjects(all_subjects)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = device == "cuda"

    if ENSEMBLE_CHECKPOINTS:
        ckpt_paths = [Path(p) for p in ENSEMBLE_CHECKPOINTS]
        models = []
        for cp in ckpt_paths:
            m = build_model(device, checkpoint_path=cp)
            m.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
            m.eval()
            models.append(m)
        print(f"Ensemble: {len(models)} checkpoints")
    else:
        ckpt_path = resolve_checkpoint()
        model = build_model(device, checkpoint_path=ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        models = [model]
        print(f"Checkpoint: {ckpt_path}")

    enhancements = []
    if USE_TTA:
        enhancements.append("TTA")
    if USE_CC_POSTPROC:
        enhancements.append("CC")
    if ENSEMBLE_CHECKPOINTS:
        enhancements.append(f"Ensemble({len(models)})")

    print(f"Experiment: {EXPERIMENT_NAME}  |  Model: {MODEL_VERSION}")
    print(f"Val subjects: {len(val_subjects)}  |  Split: {split_source}")
    print(f"Device: {device}")
    print(f"Enhancements: {' + '.join(enhancements) if enhancements else 'none'}")

    thresholds = sorted(set(float(t) for t in THRESHOLDS + [BASE_THRESHOLD]))
    threshold_results = {thr: [] for thr in thresholds}
    subject_rows = []

    for sid in tqdm(val_subjects, desc="Evaluating 3D"):
        image = np.load(str(IMG_DIR / f"{sid}.npy"))
        mask = np.load(str(MSK_DIR / f"{sid}.npy"))

        prob_acc = np.zeros_like(mask, dtype=np.float32)
        for m in models:
            prob_acc += infer_subject(m, image, PATCH_SIZE, OVERLAP, device, amp_enabled)
        pred_prob = prob_acc / len(models)

        base_dice = None
        for thr in thresholds:
            pred_bin = (pred_prob >= thr).astype(np.uint8)
            if USE_CC_POSTPROC:
                pred_bin = largest_connected_component(pred_bin)
            d = dice_3d(pred_bin, mask)
            threshold_results[thr].append(d)
            if abs(thr - BASE_THRESHOLD) < 1e-12:
                base_dice = d

        subject_rows.append({
            "subject_id": sid,
            "tumor_voxels": int(mask.sum()),
            "dice_3d": base_dice,
        })

    subject_df = pd.DataFrame(subject_rows).sort_values("subject_id").reset_index(drop=True)
    subject_df.to_csv(EVAL_DIR / "subject_metrics_3d.csv", index=False)

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
    thr_df.to_csv(EVAL_DIR / "threshold_sweep_3d.csv", index=False)

    base_arr = np.array(threshold_results[BASE_THRESHOLD])
    print(f"\n3D Dice @ threshold {BASE_THRESHOLD}:")
    print(f"  Mean:   {base_arr.mean():.4f} +/- {base_arr.std():.4f}")
    print(f"  Median: {np.median(base_arr):.4f}")
    if enhancements:
        print(f"  Enhancements: {' + '.join(enhancements)}")
    print(f"\nThreshold sweep:")
    print(thr_df.to_string(index=False))
    print(f"\nSaved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
