"""Fusion overlay visualizations for 3D U-Net v2 (FLAIR + T1gd as one RGB background).

Combines each axial slice into a single RGB image: T1gd → red, FLAIR → green (blue unused).
Ground truth, prediction, and probability heatmaps are overlaid on this fused background
instead of FLAIR alone.

Outputs
-------
  outputs/3d_unet_v2/visualizations/overlays_fusion/best/{subject_id}.png
  outputs/3d_unet_v2/visualizations/overlays_fusion/worst/{subject_id}.png
  outputs/3d_unet_v2/visualizations/overlays_fusion/random/{subject_id}.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.sliding_window import sliding_window_inference

try:
    from models.model_3d_v2 import UNet3D_V2
except ImportError:
    UNet3D_V2 = None

# ── Config (aligned with visualize_3d.py for 3d_unet_v2) ────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "3d_unet_v2"
DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_PATH = OUT_ROOT / "checkpoints" / f"best_model_{EXPERIMENT_NAME}.pt"
EVAL_CSV = OUT_ROOT / "evaluation" / "subject_metrics_3d.csv"
VAL_SUBJECTS_TXT = OUT_ROOT / "logs" / "val_subjects.txt"

VIZ_DIR = OUT_ROOT / "visualizations"
FUSION_DIR = VIZ_DIR / "overlays_fusion"
BEST_DIR = FUSION_DIR / "best"
WORST_DIR = FUSION_DIR / "worst"
RANDOM_DIR = FUSION_DIR / "random"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
for _d in (BEST_DIR, WORST_DIR, RANDOM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (128, 128, 128)
SW_OVERLAP = 0.5
THRESHOLD = 0.5

N_BEST = 10
N_WORST = 10
N_RANDOM = 10
RANDOM_SEED = 42

GT_COLOR = "#00E5FF"
PRED_COLOR = "#FF4081"


def build_model(device: str):
    if UNet3D_V2 is None:
        raise ImportError("model_3d_v2.py not found.")
    return UNet3D_V2(in_channels=2, out_channels=1, deep_supervision=True).to(device)


def best_axial_slice(mask_hwz: np.ndarray) -> int:
    areas = mask_hwz.sum(axis=(0, 1))
    return int(np.argmax(areas))


def normalize_for_display(vol: np.ndarray) -> np.ndarray:
    """Clip to [1,99] percentile and rescale to [0,1] for display only."""
    vol = np.asarray(vol, dtype=np.float32)
    fg = vol[vol > 0]
    if fg.size == 0:
        return np.zeros_like(vol)
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)


def fuse_flair_t1gd_rgb(flair_sl: np.ndarray, t1gd_sl: np.ndarray) -> np.ndarray:
    """Stack modalities into RGB: R=T1gd, G=FLAIR, B=0 (per-slice contrast)."""
    f = normalize_for_display(flair_sl)
    t = normalize_for_display(t1gd_sl)
    h, w = f.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = t
    rgb[:, :, 1] = f
    return np.clip(rgb, 0, 1)


def overlay_rgb(ax, rgb: np.ndarray, mask_slice: np.ndarray, color: str, alpha: float = 0.45):
    """Overlay a binary mask on an RGB slice."""
    ax.imshow(rgb)
    rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    c = matplotlib.colors.to_rgba(color)
    rgba[mask_slice > 0] = [c[0], c[1], c[2], alpha]
    ax.imshow(rgba, interpolation="none")
    ax.axis("off")


def _assign_groups(eval_df: pd.DataFrame, val_subjects: list):
    rng = np.random.default_rng(RANDOM_SEED)
    df = (
        eval_df[eval_df["subject_id"].isin(val_subjects)]
        .sort_values("dice_3d", ascending=False)
        .reset_index(drop=True)
    )
    best_sids = df.head(N_BEST)["subject_id"].tolist()
    worst_sids = df.tail(N_WORST)["subject_id"].tolist()
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


def _render_one(sid: str, dice_val: float, out_dir: Path, group_label: str, model, device: str):
    image = np.load(str(IMG_DIR / f"{sid}.npy"))
    mask = np.load(str(MSK_DIR / f"{sid}.npy"))

    pred_prob = sliding_window_inference(
        model,
        image,
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
    fused = fuse_flair_t1gd_rgb(flair_sl, t1gd_sl)
    gt_sl = mask[:, :, z]
    pred_sl = pred_bin[:, :, z]
    prob_sl = pred_prob[:, :, z]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("#111111")

    axes[0].imshow(fused)
    axes[0].set_title("FLAIR + T1gd\n(R=T1gd, G=FLAIR)", color="white", fontsize=10)
    axes[0].axis("off")

    overlay_rgb(axes[1], fused, gt_sl, GT_COLOR)
    axes[1].set_title("GT mask", color="white", fontsize=11)

    overlay_rgb(axes[2], fused, pred_sl, PRED_COLOR)
    axes[2].set_title("Prediction", color="white", fontsize=11)

    axes[3].imshow(fused)
    prob_img = axes[3].imshow(prob_sl, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title("Probability", color="white", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(prob_img, ax=axes[3], fraction=0.046, pad=0.04)

    group_colors = {"best": "#69FF47", "worst": "#FF4081", "random": "#FFD700"}
    title = (
        f"[{group_label.upper()}]  {sid}  |  axial z={z}"
        f"  |  3D Dice = {dice_val:.4f}"
    )
    fig.suptitle(
        title,
        color=group_colors.get(group_label, "white"),
        fontsize=13,
        y=1.02,
    )

    gt_patch = mpatches.Patch(color=GT_COLOR, label="Ground truth", alpha=0.7)
    pred_patch = mpatches.Patch(color=PRED_COLOR, label="Prediction", alpha=0.7)
    fig.legend(
        handles=[gt_patch, pred_patch],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(
        out_dir / f"{sid}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def make_fusion_overlays(model, device: str, val_subjects: list, eval_df: pd.DataFrame | None):
    if eval_df is None or "dice_3d" not in eval_df.columns:
        print("No eval CSV with dice_3d — skipping fusion overlays.")
        return

    assignment = _assign_groups(eval_df, val_subjects)
    counts = {"best": 0, "worst": 0, "random": 0}
    for sid, (dice_val, out_dir, group) in tqdm(
        assignment.items(), desc="Fusion overlays", total=len(assignment)
    ):
        _render_one(sid, dice_val, out_dir, group, model, device)
        counts[group] += 1

    print(
        f"Fusion overlays — best: {counts['best']}  worst: {counts['worst']}  "
        f"random: {counts['random']}"
    )
    print(f"  {BEST_DIR}")
    print(f"  {WORST_DIR}")
    print(f"  {RANDOM_DIR}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Experiment: {EXPERIMENT_NAME}  |  Fusion overlays (FLAIR+T1gd)")
    print(f"Checkpoint: {CKPT_PATH}")

    model = build_model(device)
    model.load_state_dict(
        torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    )
    model.eval()

    val_subjects = [
        line.strip()
        for line in VAL_SUBJECTS_TXT.read_text().splitlines()
        if line.strip()
    ]
    print(f"Val subjects: {len(val_subjects)}")

    eval_df = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None
    make_fusion_overlays(model, device, val_subjects, eval_df)
    print(f"\nDone. Figures under: {FUSION_DIR}")


if __name__ == "__main__":
    main()
