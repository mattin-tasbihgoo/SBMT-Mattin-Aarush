from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorSliceDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# Experiment paths
# --------------------------------------------------
DATA_EXPERIMENT_NAME = "runtime_flair_t1gd_25d5"
EXPERIMENT_NAME = "runtime_flair_t1gd_25d5_smalllesion_bce_tversky"

DATA_ROOT = PROJECT_ROOT / "data" / DATA_EXPERIMENT_NAME
META_CSV = DATA_ROOT / "metadata.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Reference split
# Reuse the frozen baseline split when available so
# experiments stay directly comparable.
# --------------------------------------------------
REFERENCE_SPLIT_RUN_NAME = "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"
REFERENCE_SPLIT_LOG_DIR = PROJECT_ROOT / "outputs" / REFERENCE_SPLIT_RUN_NAME / "logs"
REFERENCE_TRAIN_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "train_subjects.txt"
REFERENCE_VAL_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "val_subjects.txt"

# --------------------------------------------------
# Tunable settings
# --------------------------------------------------
IN_CHANNELS = 10
BATCH_SIZE = 48
NUM_WORKERS = 2
EPOCHS = 10
LR = 1e-3
VAL_FRAC = 0.20
RANDOM_SEED = 42
POSTFIX_EVERY = 20
PRELOAD_SUBJECTS = True

# Small-lesion-aware sampling
USE_SMALL_LESION_OVERSAMPLING = True
SMALL_LESION_MULTIPLIER = 2

# Loss
LOSS_NAME = "BCE + Tversky"
LOSS_BCE_WEIGHT = 0.5
LOSS_TVERSKY_WEIGHT = 0.5
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7

# Model selection
MODEL_SELECTION_METRIC = "val_pos_dice"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


bce = nn.BCEWithLogitsLoss()


def tversky_loss(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, eps=1e-6):
    probs = torch.sigmoid(logits)

    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def combined_loss(logits, targets):
    loss_bce = bce(logits, targets)
    loss_tversky = tversky_loss(logits, targets)
    return LOSS_BCE_WEIGHT * loss_bce + LOSS_TVERSKY_WEIGHT * loss_tversky


def dice_scores(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (union + eps)


def add_small_lesion_oversampling(pos_train: pd.DataFrame) -> pd.DataFrame:
    pos_train = pos_train.copy().reset_index(drop=True)

    if "lesion_area" not in pos_train.columns:
        raise RuntimeError("metadata.csv is missing lesion_area column.")

    if len(pos_train) < 3:
        pos_train["lesion_size_group"] = "small"
        return pos_train

    pos_train = pos_train.sort_values("lesion_area", ascending=True).reset_index(drop=True)

    n = len(pos_train)
    small_end = n // 3
    medium_end = 2 * n // 3

    groups = np.array(["large"] * n, dtype=object)
    groups[:small_end] = "small"
    groups[small_end:medium_end] = "medium"

    pos_train["lesion_size_group"] = groups

    if USE_SMALL_LESION_OVERSAMPLING:
        small_df = pos_train[pos_train["lesion_size_group"] == "small"].copy()
        extras = [small_df.copy() for _ in range(max(0, SMALL_LESION_MULTIPLIER - 1))]
        if extras:
            pos_train = pd.concat([pos_train] + extras, axis=0).reset_index(drop=True)

    return pos_train


def load_subject_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_subject_split(df: pd.DataFrame):
    all_subjects = set(df["subject_id"].unique().tolist())

    if REFERENCE_TRAIN_SUBJECTS_TXT.exists() and REFERENCE_VAL_SUBJECTS_TXT.exists():
        train_subjects = load_subject_list(REFERENCE_TRAIN_SUBJECTS_TXT)
        val_subjects = load_subject_list(REFERENCE_VAL_SUBJECTS_TXT)

        missing_train = sorted(set(train_subjects) - all_subjects)
        missing_val = sorted(set(val_subjects) - all_subjects)
        overlap = sorted(set(train_subjects) & set(val_subjects))

        if missing_train or missing_val:
            missing_text = []
            if missing_train:
                missing_text.append("Missing train subjects:\n" + "\n".join(missing_train[:20]))
            if missing_val:
                missing_text.append("Missing val subjects:\n" + "\n".join(missing_val[:20]))
            raise RuntimeError("Reference split subjects were not found in metadata.\n" + "\n\n".join(missing_text))

        if overlap:
            raise RuntimeError(
                "Reference split is invalid: some subjects appear in both train and val.\n"
                + "\n".join(overlap[:20])
            )

        return train_subjects, val_subjects, f"loaded from {REFERENCE_SPLIT_LOG_DIR}"

    subjects = sorted(all_subjects)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(subjects)

    n_val = max(1, int(len(subjects) * VAL_FRAC))
    val_subjects = subjects[:n_val]
    train_subjects = subjects[n_val:]

    return train_subjects, val_subjects, "deterministic RANDOM_SEED/VAL_FRAC fallback"


def write_run_config(split_source: str):
    config_lines = [
        f"experiment_name: {EXPERIMENT_NAME}",
        f"data_experiment_name: {DATA_EXPERIMENT_NAME}",
        f"metadata_csv: {META_CSV}",
        f"reference_split_run_name: {REFERENCE_SPLIT_RUN_NAME}",
        f"split_source: {split_source}",
        f"in_channels: {IN_CHANNELS}",
        f"batch_size: {BATCH_SIZE}",
        f"num_workers: {NUM_WORKERS}",
        f"epochs: {EPOCHS}",
        f"lr: {LR}",
        f"random_seed: {RANDOM_SEED}",
        f"preload_subjects: {PRELOAD_SUBJECTS}",
        f"loss_name: {LOSS_NAME}",
        f"loss_bce_weight: {LOSS_BCE_WEIGHT}",
        f"loss_tversky_weight: {LOSS_TVERSKY_WEIGHT}",
        f"tversky_alpha: {TVERSKY_ALPHA}",
        f"tversky_beta: {TVERSKY_BETA}",
        f"use_small_lesion_oversampling: {USE_SMALL_LESION_OVERSAMPLING}",
        f"small_lesion_multiplier: {SMALL_LESION_MULTIPLIER}",
        f"model_selection_metric: {MODEL_SELECTION_METRIC}",
    ]
    (LOG_DIR / "run_config.txt").write_text("\n".join(config_lines))


def main():
    set_seed(RANDOM_SEED)

    if not META_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {META_CSV}\n"
            "Run build_runtime_metadata_flair_t1gd_25d5.py first."
        )

    df = pd.read_csv(META_CSV)
    if df.empty:
        raise RuntimeError("metadata.csv is empty. Run metadata building first.")

    train_subjects, val_subjects, split_source = resolve_subject_split(df)

    train_df = df[df["subject_id"].isin(train_subjects)].copy()
    val_df = df[df["subject_id"].isin(val_subjects)].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError("Train or validation dataframe is empty after subject split.")

    pos_train = train_df[train_df["has_lesion"] == 1].copy()
    neg_train = train_df[train_df["has_lesion"] == 0].copy()

    if len(pos_train) == 0:
        raise RuntimeError("No positive training slices found.")

    pos_train = add_small_lesion_oversampling(pos_train)

    if len(neg_train) > 0:
        neg_train = neg_train.sample(
            n=min(len(neg_train), 2 * len(pos_train)),
            random_state=RANDOM_SEED,
        )

    train_df = (
        pd.concat([pos_train, neg_train], axis=0)
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    print("Experiment:", EXPERIMENT_NAME)
    print("Data experiment:", DATA_EXPERIMENT_NAME)
    print("Metadata:", META_CSV)
    print("Split source:", split_source)
    print("Train subjects:", len(train_subjects))
    print("Val subjects:", len(val_subjects))
    print("Train slices:", len(train_df))
    print("Val slices:", len(val_df))
    print("Train positives after oversampling:", int((train_df["has_lesion"] == 1).sum()))
    print("Val positives:", int(val_df["has_lesion"].sum()))
    print("Loss:", LOSS_NAME)
    print("Tversky alpha:", TVERSKY_ALPHA)
    print("Tversky beta:", TVERSKY_BETA)
    print("Small-lesion oversampling enabled:", USE_SMALL_LESION_OVERSAMPLING)
    print("Small-lesion multiplier:", SMALL_LESION_MULTIPLIER)
    print("Preload subjects:", PRELOAD_SUBJECTS)

    if "lesion_size_group" in pos_train.columns:
        print("\nPositive training size groups after oversampling:")
        print(pos_train["lesion_size_group"].value_counts())

        print("\nPositive lesion area summary:")
        print(pos_train["lesion_area"].describe())

    (LOG_DIR / "train_subjects.txt").write_text("\n".join(train_subjects))
    (LOG_DIR / "val_subjects.txt").write_text("\n".join(val_subjects))
    write_run_config(split_source)

    train_ds = BrainTumorSliceDataset(train_df, augment=True, preload_subjects=PRELOAD_SUBJECTS)
    val_ds = BrainTumorSliceDataset(val_df, augment=False, preload_subjects=PRELOAD_SUBJECTS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = build_model().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device=amp_device, enabled=(device == "cuda"))

    best_val_metric = -1.0
    history = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_seen = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [train]",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

        for batch_idx, (x, y) in enumerate(train_pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=amp_device, enabled=(device == "cuda")):
                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_seen += bs

            if batch_idx % POSTFIX_EVERY == 0 or batch_idx == len(train_loader):
                postfix = {"loss": f"{train_loss_sum / max(1, train_seen):.4f}"}
                if device == "cuda":
                    postfix["gpu_mem_gb"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
                train_pbar.set_postfix(postfix)

        train_loss = train_loss_sum / max(1, train_seen)

        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_seen = 0
        val_pos_dice_sum = 0.0
        val_pos_seen = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [val]",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_pbar, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(device_type=amp_device, enabled=(device == "cuda")):
                    logits = model(x)
                    loss = combined_loss(logits, y)

                batch_dice = dice_scores(logits, y)

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_dice_sum += batch_dice.sum().item()
                val_seen += bs

                pos_mask = y.sum(dim=(1, 2, 3)) > 0
                if pos_mask.any():
                    val_pos_dice_sum += batch_dice[pos_mask].sum().item()
                    val_pos_seen += int(pos_mask.sum().item())

                if batch_idx % POSTFIX_EVERY == 0 or batch_idx == len(val_loader):
                    val_pbar.set_postfix({
                        "loss": f"{val_loss_sum / max(1, val_seen):.4f}",
                        "dice": f"{val_dice_sum / max(1, val_seen):.4f}",
                        "pos_dice": f"{val_pos_dice_sum / max(1, val_pos_seen):.4f}",
                    })

        val_loss = val_loss_sum / max(1, val_seen)
        val_dice = val_dice_sum / max(1, val_seen)
        val_pos_dice = val_pos_dice_sum / max(1, val_pos_seen)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_pos_dice": val_pos_dice,
            "epoch_time_sec": epoch_time,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "lr": LR,
            "in_channels": IN_CHANNELS,
            "preload_subjects": PRELOAD_SUBJECTS,
            "loss_name": LOSS_NAME,
            "loss_bce_weight": LOSS_BCE_WEIGHT,
            "loss_tversky_weight": LOSS_TVERSKY_WEIGHT,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
            "use_small_lesion_oversampling": USE_SMALL_LESION_OVERSAMPLING,
            "small_lesion_multiplier": SMALL_LESION_MULTIPLIER,
            "model_selection_metric": MODEL_SELECTION_METRIC,
        })

        pd.DataFrame(history).to_csv(LOG_DIR / "training_log.csv", index=False)
        torch.save(model.state_dict(), CKPT_DIR / f"latest_model_{EXPERIMENT_NAME}.pt")

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_pos_dice={val_pos_dice:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        current_metric = val_pos_dice
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt")
            print(f"Saved best model by {MODEL_SELECTION_METRIC}.")

    print(f"Saved training log to: {LOG_DIR / 'training_log.csv'}")
    print(f"Best checkpoint: {CKPT_DIR / f'best_model_{EXPERIMENT_NAME}.pt'}")


if __name__ == "__main__":
    main()
