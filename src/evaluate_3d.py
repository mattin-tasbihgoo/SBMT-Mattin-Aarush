"""Evaluate a trained 3D U-Net on validation subjects.

Loads each subject's full volume, runs sliding-window inference,
computes per-subject 3D Dice, and writes summary CSVs.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_3d import UNet3D
from sliding_window import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "3d_unet_v1"
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

PATCH_SIZE = (128, 128, 128)
OVERLAP = 0.5
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
BASE_THRESHOLD = 0.50
RANDOM_SEED = 42
VAL_FRAC = 0.20


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


def main():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(META_CSV)
    all_subjects = sorted(df["subject_id"].unique().tolist())
    val_subjects, split_source = resolve_val_subjects(all_subjects)

    ckpt_path = resolve_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = device

    print("Experiment:", EXPERIMENT_NAME)
    print("Checkpoint:", ckpt_path)
    print("Val subjects:", len(val_subjects))
    print("Split source:", split_source)
    print("Device:", device)

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    thresholds = sorted(set(float(t) for t in THRESHOLDS + [BASE_THRESHOLD]))
    threshold_results = {thr: [] for thr in thresholds}

    subject_rows = []

    for sid in tqdm(val_subjects, desc="Evaluating 3D"):
        image = np.load(str(IMG_DIR / f"{sid}.npy"))  # (2,H,W,D)
        mask = np.load(str(MSK_DIR / f"{sid}.npy"))    # (H,W,D)

        pred_prob = sliding_window_inference(
            model, image,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            device=device,
            amp_enabled=(device == "cuda"),
        )

        base_dice = None
        for thr in thresholds:
            pred_bin = (pred_prob >= thr).astype(np.uint8)
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
    print(f"\nThreshold sweep:")
    print(thr_df.to_string(index=False))
    print(f"\nSaved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
