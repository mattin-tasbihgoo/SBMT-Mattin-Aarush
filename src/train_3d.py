"""Train a 3D U-Net for volumetric brain tumor segmentation.

Reuses subject split, loss design, logging, and checkpoint conventions
from the 2.5D train.py but operates on (128,128,128) patches extracted
by BrainTumorDataset3D, with sliding-window validation.
"""

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_3d import BrainTumorDataset3D
from model_3d import UNet3D
from sliding_window import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data ─────────────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
META_CSV = DATA_ROOT / "metadata.csv"

# ── Experiment ───────────────────────────────────────────────────────
EXPERIMENT_NAME = "3d_unet_v1"
OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference split (reuse 2.5D when available) ─────────────────────
REFERENCE_SPLIT_RUN_NAME = "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"
REFERENCE_SPLIT_LOG_DIR = PROJECT_ROOT / "outputs" / REFERENCE_SPLIT_RUN_NAME / "logs"
REFERENCE_TRAIN_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "train_subjects.txt"
REFERENCE_VAL_SUBJECTS_TXT = REFERENCE_SPLIT_LOG_DIR / "val_subjects.txt"

# ── Hyperparameters ──────────────────────────────────────────────────
PATCH_SIZE = (128, 128, 128)
PATCHES_PER_SUBJECT = 4
BATCH_SIZE = 4
NUM_WORKERS = 8
EPOCHS = 50
LR = 1e-4
VAL_FRAC = 0.20
RANDOM_SEED = 42
GRAD_CLIP_NORM = 1.0
POSTFIX_EVERY = 5

# Loss
LOSS_NAME = "BCE + Dice"
LOSS_BCE_WEIGHT = 0.5
LOSS_DICE_WEIGHT = 0.5

# Sliding-window validation
SW_OVERLAP = 0.5
SW_THRESHOLD = 0.5

# How many subjects to run sliding-window eval on each epoch
# (full eval is expensive; set to 0 to skip per-epoch SW eval)
VAL_SW_SUBJECTS = 5

# ─────────────────────────────────────────────────────────────────────

bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3, 4))
    union = probs.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def combined_loss(logits, targets):
    return LOSS_BCE_WEIGHT * bce(logits, targets) + LOSS_DICE_WEIGHT * dice_loss(logits, targets)


def dice_scores(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(1, 2, 3, 4))
    union = preds.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
    return (2.0 * inter + eps) / (union + eps)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_list(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def resolve_subject_split(all_subjects: set[str]):
    if REFERENCE_TRAIN_SUBJECTS_TXT.exists() and REFERENCE_VAL_SUBJECTS_TXT.exists():
        train_subjects = load_subject_list(REFERENCE_TRAIN_SUBJECTS_TXT)
        val_subjects = load_subject_list(REFERENCE_VAL_SUBJECTS_TXT)

        missing = sorted((set(train_subjects) | set(val_subjects)) - all_subjects)
        overlap = sorted(set(train_subjects) & set(val_subjects))
        if missing:
            raise RuntimeError("Reference split subjects missing:\n" + "\n".join(missing[:20]))
        if overlap:
            raise RuntimeError("Split overlap:\n" + "\n".join(overlap[:20]))

        return train_subjects, val_subjects, f"loaded from {REFERENCE_SPLIT_LOG_DIR}"

    subjects = sorted(all_subjects)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * VAL_FRAC))
    return subjects[n_val:], subjects[:n_val], "deterministic RANDOM_SEED/VAL_FRAC fallback"


def dice_3d_np(pred_prob, mask, threshold=0.5, eps=1e-6):
    pred = (pred_prob >= threshold).astype(bool)
    target = mask.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def main():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(META_CSV)
    all_subjects = set(df["subject_id"].unique().tolist())
    train_subjects, val_subjects, split_source = resolve_subject_split(all_subjects)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Split source: {split_source}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Patches/subject: {PATCHES_PER_SUBJECT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Loss: {LOSS_NAME}")

    (LOG_DIR / "train_subjects.txt").write_text("\n".join(train_subjects))
    (LOG_DIR / "val_subjects.txt").write_text("\n".join(val_subjects))

    train_ds = BrainTumorDataset3D(
        train_subjects, str(IMG_DIR), str(MSK_DIR),
        patch_size=PATCH_SIZE,
        patches_per_subject=PATCHES_PER_SUBJECT,
        mode="train",
        augment=True,
    )
    val_ds = BrainTumorDataset3D(
        val_subjects, str(IMG_DIR), str(MSK_DIR),
        patch_size=PATCH_SIZE,
        patches_per_subject=PATCHES_PER_SUBJECT,
        mode="train",
        augment=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(device=amp_device, enabled=(device == "cuda"))

    best_val_metric = -1.0
    history = []

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]",
                     dynamic_ncols=True, leave=True, mininterval=0.5)
        for batch_idx, (x, y, _sids) in enumerate(pbar, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device, enabled=(device == "cuda")):
                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_dice_sum += dice_scores(logits, y).sum().item()
            train_seen += bs

            if batch_idx % POSTFIX_EVERY == 0 or batch_idx == len(train_loader):
                postfix = {
                    "loss": f"{train_loss_sum / max(1, train_seen):.4f}",
                    "dice": f"{train_dice_sum / max(1, train_seen):.4f}",
                }
                if device == "cuda":
                    postfix["mem_gb"] = f"{torch.cuda.memory_allocated()/1e9:.1f}"
                pbar.set_postfix(postfix)

        train_loss = train_loss_sum / max(1, train_seen)
        train_dice = train_dice_sum / max(1, train_seen)

        # ── Patch-based val loss ─────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_seen = 0

        vpbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]",
                      dynamic_ncols=True, leave=True, mininterval=0.5)
        with torch.no_grad():
            for batch_idx, (x, y, _sids) in enumerate(vpbar, 1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(device_type=amp_device, enabled=(device == "cuda")):
                    logits = model(x)
                    loss = combined_loss(logits, y)

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_dice_sum += dice_scores(logits, y).sum().item()
                val_seen += bs

                if batch_idx % POSTFIX_EVERY == 0 or batch_idx == len(val_loader):
                    vpbar.set_postfix({
                        "loss": f"{val_loss_sum / max(1, val_seen):.4f}",
                        "dice": f"{val_dice_sum / max(1, val_seen):.4f}",
                    })

        val_loss = val_loss_sum / max(1, val_seen)
        val_dice = val_dice_sum / max(1, val_seen)

        # ── Sliding-window val on a subset ───────────────────────────
        sw_dice_mean = float("nan")
        if VAL_SW_SUBJECTS > 0:
            sw_subs = val_subjects[:VAL_SW_SUBJECTS]
            sw_dices = []
            for sid in tqdm(sw_subs, desc=f"Epoch {epoch+1}/{EPOCHS} [SW val]",
                            dynamic_ncols=True, leave=False):
                image = np.load(str(IMG_DIR / f"{sid}.npy"))
                mask = np.load(str(MSK_DIR / f"{sid}.npy"))
                pred_prob = sliding_window_inference(
                    model, image, patch_size=PATCH_SIZE, overlap=SW_OVERLAP,
                    device=device, amp_enabled=(device == "cuda"),
                )
                sw_dices.append(dice_3d_np(pred_prob, mask, SW_THRESHOLD))
            sw_dice_mean = float(np.mean(sw_dices))

        scheduler.step()
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "sw_val_dice": sw_dice_mean,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(LOG_DIR / "training_log.csv", index=False)
        torch.save(model.state_dict(), CKPT_DIR / f"latest_model_{EXPERIMENT_NAME}.pt")

        current_metric = sw_dice_mean if not np.isnan(sw_dice_mean) else val_dice
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt")
            print(f"  ** Saved best model (metric={current_metric:.4f})")

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
            f"sw_val_dice={sw_dice_mean:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={epoch_time:.0f}s"
        )

    print(f"\nTraining complete. Log: {LOG_DIR / 'training_log.csv'}")
    print(f"Best checkpoint: {CKPT_DIR / f'best_model_{EXPERIMENT_NAME}.pt'}")


if __name__ == "__main__":
    main()
