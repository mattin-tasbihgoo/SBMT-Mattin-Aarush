"""Generate poster overlay visualizations from a trained 3D U-Net.

Overlays are split into three folders based on per-subject 3D Dice:
  best/   — top N subjects by Dice
  worst/  — bottom N subjects by Dice
  random/ — N randomly sampled from the middle tier

Outputs
-------
  outputs/3d_unet_v1/visualizations/overlays/best/{subject_id}.png
  outputs/3d_unet_v1/visualizations/overlays/worst/{subject_id}.png
  outputs/3d_unet_v1/visualizations/overlays/random/{subject_id}.png
  outputs/3d_unet_v1/visualizations/training_curves.png
  outputs/3d_unet_v1/visualizations/dice_distribution.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_3d import UNet3D
from sliding_window import sliding_window_inference

# ── Config ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "3d_unet_v1"
DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_PATH = OUT_ROOT / "checkpoints" / f"best_model_{EXPERIMENT_NAME}.pt"
LOG_CSV = OUT_ROOT / "logs" / "training_log.csv"
EVAL_CSV = OUT_ROOT / "evaluation" / "subject_metrics_3d.csv"
VAL_SUBJECTS_TXT = OUT_ROOT / "logs" / "val_subjects.txt"

VIZ_DIR = OUT_ROOT / "visualizations"
OVERLAY_DIR = VIZ_DIR / "overlays"
BEST_DIR = OVERLAY_DIR / "best"
WORST_DIR = OVERLAY_DIR / "worst"
RANDOM_DIR = OVERLAY_DIR / "random"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
for _d in (BEST_DIR, WORST_DIR, RANDOM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (128, 128, 128)
SW_OVERLAP = 0.5
THRESHOLD = 0.5

# Number of subjects in each category
N_BEST = 10
N_WORST = 10
N_RANDOM = 10

RANDOM_SEED = 42

# Poster palette
FLAIR_CMAP = "gray"
T1GD_CMAP = "gray"
GT_COLOR = "#00E5FF"    # cyan
PRED_COLOR = "#FF4081"  # pink-red


# ── Helpers ───────────────────────────────────────────────────────────

def overlay(ax, img_slice, mask_slice, color, alpha=0.45):
    """Overlay a binary mask on a grayscale slice."""
    ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    c = matplotlib.colors.to_rgba(color)
    rgba[mask_slice > 0] = [c[0], c[1], c[2], alpha]
    ax.imshow(rgba, interpolation="none")
    ax.axis("off")


def best_axial_slice(mask_hwz):
    """Return the axial (z) index with the largest lesion area."""
    areas = mask_hwz.sum(axis=(0, 1))
    return int(np.argmax(areas))


def normalize_for_display(vol):
    """Clip to [1,99] percentile and rescale to [0,1] for display only."""
    vol = np.asarray(vol, dtype=np.float32)
    fg = vol[vol > 0]
    if fg.size == 0:
        return np.zeros_like(vol)
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)


# ── Main ──────────────────────────────────────────────────────────────

def _assign_groups(eval_df, val_subjects):
    """Return dicts: {subject_id: (dice, output_dir, group_label)}."""
    rng = np.random.default_rng(RANDOM_SEED)

    df = (
        eval_df[eval_df["subject_id"].isin(val_subjects)]
        .sort_values("dice_3d", ascending=False)
        .reset_index(drop=True)
    )

    best_sids = df.head(N_BEST)["subject_id"].tolist()
    worst_sids = df.tail(N_WORST)["subject_id"].tolist()

    # Middle tier = everything not in best or worst
    used = set(best_sids) | set(worst_sids)
    middle = df[~df["subject_id"].isin(used)]["subject_id"].tolist()
    n_rand = min(N_RANDOM, len(middle))
    random_sids = rng.choice(middle, size=n_rand, replace=False).tolist()

    assignment = {}
    for sid in best_sids:
        row = df[df["subject_id"] == sid].iloc[0]
        assignment[sid] = (float(row["dice_3d"]), BEST_DIR, "best")
    for sid in worst_sids:
        row = df[df["subject_id"] == sid].iloc[0]
        assignment[sid] = (float(row["dice_3d"]), WORST_DIR, "worst")
    for sid in random_sids:
        row = df[df["subject_id"] == sid].iloc[0]
        assignment[sid] = (float(row["dice_3d"]), RANDOM_DIR, "random")

    return assignment


def _render_one(sid, dice_val, out_dir, group_label, model, device):
    """Run inference and save a 5-panel overlay for one subject."""
    image = np.load(str(IMG_DIR / f"{sid}.npy"))  # (2, H, W, D)
    mask = np.load(str(MSK_DIR / f"{sid}.npy"))    # (H, W, D)

    pred_prob = sliding_window_inference(
        model, image,
        patch_size=PATCH_SIZE,
        overlap=SW_OVERLAP,
        device=device,
        amp_enabled=(device == "cuda"),
    )
    pred_bin = (pred_prob >= THRESHOLD).astype(np.uint8)

    flair = image[0]
    t1gd = image[1]
    z = best_axial_slice(mask)

    flair_sl = flair[:, :, z]
    t1gd_sl = t1gd[:, :, z]
    gt_sl = mask[:, :, z]
    pred_sl = pred_bin[:, :, z]
    prob_sl = pred_prob[:, :, z]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("#111111")

    axes[0].imshow(flair_sl, cmap=FLAIR_CMAP, vmin=0, vmax=1)
    axes[0].set_title("FLAIR", color="white", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(t1gd_sl, cmap=T1GD_CMAP, vmin=0, vmax=1)
    axes[1].set_title("T1gd", color="white", fontsize=11)
    axes[1].axis("off")

    overlay(axes[2], flair_sl, gt_sl, GT_COLOR)
    axes[2].set_title("GT mask", color="white", fontsize=11)

    overlay(axes[3], flair_sl, pred_sl, PRED_COLOR)
    axes[3].set_title("Prediction", color="white", fontsize=11)

    axes[4].imshow(flair_sl, cmap="gray", vmin=0, vmax=1)
    prob_img = axes[4].imshow(prob_sl, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[4].set_title("Probability", color="white", fontsize=11)
    axes[4].axis("off")
    plt.colorbar(prob_img, ax=axes[4], fraction=0.046, pad=0.04)

    group_colors = {"best": "#69FF47", "worst": "#FF4081", "random": "#FFD700"}
    title = (
        f"[{group_label.upper()}]  {sid}  |  axial z={z}"
        f"  |  3D Dice = {dice_val:.4f}"
    )
    fig.suptitle(title, color=group_colors.get(group_label, "white"),
                 fontsize=13, y=1.01)

    gt_patch = mpatches.Patch(color=GT_COLOR, label="Ground truth", alpha=0.7)
    pred_patch = mpatches.Patch(color=PRED_COLOR, label="Prediction", alpha=0.7)
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, frameon=False, fontsize=10,
               labelcolor="white", bbox_to_anchor=(0.5, -0.04))

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / f"{sid}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def make_overlay_panels(model, device, val_subjects, eval_df):
    """Generate overlay PNGs split into best / worst / random sub-folders."""
    if eval_df is None or "dice_3d" not in eval_df.columns:
        print("  No eval CSV found — skipping overlay panels.")
        return

    assignment = _assign_groups(eval_df, val_subjects)

    counts = {"best": 0, "worst": 0, "random": 0}
    for sid, (dice_val, out_dir, group) in tqdm(
        assignment.items(), desc="Generating overlays", total=len(assignment)
    ):
        _render_one(sid, dice_val, out_dir, group, model, device)
        counts[group] += 1

    print(
        f"Overlay panels saved — "
        f"best: {counts['best']}  worst: {counts['worst']}  random: {counts['random']}"
    )
    print(f"  {BEST_DIR}")
    print(f"  {WORST_DIR}")
    print(f"  {RANDOM_DIR}")


def make_training_curves(log_csv):
    df = pd.read_csv(log_csv)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Loss
    axes[0].plot(epochs, df["train_loss"], color="#FF4081", linewidth=2, label="Train loss")
    axes[0].plot(epochs, df["val_loss"], color="#00E5FF", linewidth=2, label="Val loss (patch)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE + Dice)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(frameon=False, labelcolor="white")
    axes[0].grid(True, color="#333333", linewidth=0.5)

    # Dice
    axes[1].plot(epochs, df["train_dice"], color="#FF4081", linewidth=2, label="Train Dice (patch)")
    axes[1].plot(epochs, df["val_dice"], color="#00E5FF", linewidth=2, label="Val Dice (patch)")
    if "sw_val_dice" in df.columns:
        axes[1].plot(epochs, df["sw_val_dice"], color="#69FF47", linewidth=2.5,
                     linestyle="--", label="SW Val Dice (3D, 10 subjects)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(frameon=False, labelcolor="white")
    axes[1].grid(True, color="#333333", linewidth=0.5)

    best_epoch = int(df.loc[df["sw_val_dice"].idxmax(), "epoch"])
    best_dice = float(df["sw_val_dice"].max())
    axes[1].axvline(best_epoch, color="#FFD700", linewidth=1.2, linestyle=":")
    axes[1].annotate(
        f"Best\n{best_dice:.4f}\n(ep {best_epoch})",
        xy=(best_epoch, best_dice),
        xytext=(best_epoch + 2, best_dice - 0.04),
        color="#FFD700", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.2),
    )

    fig.suptitle("3D U-Net Training — MSD Task01 BrainTumour", color="white", fontsize=14)
    plt.tight_layout()
    out = VIZ_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved training curves to: {out}")


def make_dice_distribution(eval_csv):
    df = pd.read_csv(eval_csv).sort_values("dice_3d")
    dices = df["dice_3d"].values
    mean_d = dices.mean()
    median_d = np.median(dices)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1a1a1a")

    colors = ["#FF4081" if d < 0.80 else "#00E5FF" for d in dices]
    ax.barh(range(len(dices)), dices, color=colors, edgecolor="none", height=0.8)
    ax.axvline(mean_d, color="#FFD700", linewidth=1.5, linestyle="--", label=f"Mean {mean_d:.3f}")
    ax.axvline(median_d, color="#69FF47", linewidth=1.5, linestyle=":", label=f"Median {median_d:.3f}")
    ax.axvline(0.8, color="#888888", linewidth=1.0, linestyle=":", alpha=0.6)

    ax.set_yticks(range(len(dices)))
    ax.set_yticklabels(df["subject_id"].values, fontsize=6, color="white")
    ax.set_xlabel("3D Dice (threshold = 0.5)", color="white")
    ax.set_title(f"Per-Subject 3D Dice — {len(dices)} Validation Subjects",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.set_xlim(0.4, 1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(frameon=False, labelcolor="white", fontsize=10)

    # Mark the worst subject
    worst_idx = int(df["dice_3d"].idxmin() - df.index[0])  # position in sorted df
    worst_sid = df.iloc[0]["subject_id"]
    worst_d = df.iloc[0]["dice_3d"]
    ax.annotate(f"{worst_sid}\n{worst_d:.3f}",
                xy=(worst_d, 0),
                xytext=(worst_d + 0.05, 4),
                color="#FF4081", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#FF4081", lw=1))

    plt.tight_layout()
    out = VIZ_DIR / "dice_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved Dice distribution to: {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CKPT_PATH}")

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(str(CKPT_PATH), map_location=device))
    model.eval()

    val_subjects = [
        l.strip()
        for l in VAL_SUBJECTS_TXT.read_text().splitlines()
        if l.strip()
    ]
    print(f"Val subjects: {len(val_subjects)}")

    eval_df = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None

    print("\n[1/3] Training curves")
    make_training_curves(LOG_CSV)

    print("\n[2/3] Dice distribution")
    make_dice_distribution(EVAL_CSV)

    print("\n[3/3] Overlay panels")
    make_overlay_panels(model, device, val_subjects, eval_df)

    print(f"\nAll figures saved under: {VIZ_DIR}")


if __name__ == "__main__":
    main()
