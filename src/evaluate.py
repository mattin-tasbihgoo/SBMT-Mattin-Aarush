from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorSliceDataset

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    ndi = None
    HAVE_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# Match the CURRENT successful training run
# --------------------------------------------------
DATA_EXPERIMENT_NAME = "runtime_flair_t1gd_25d5"
EXPERIMENT_NAME = "runtime_flair_t1gd_25d5_smalllesion"

DATA_ROOT = PROJECT_ROOT / "data" / DATA_EXPERIMENT_NAME
META_CSV = DATA_ROOT / "metadata.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"
EVAL_DIR = OUT_ROOT / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

VAL_SUBJECTS_TXT = LOG_DIR / "val_subjects.txt"
BEST_CKPT_PATH = CKPT_DIR / f"best_model_{EXPERIMENT_NAME}.pt"
LATEST_CKPT_PATH = CKPT_DIR / f"latest_model_{EXPERIMENT_NAME}.pt"

# --------------------------------------------------
# Settings
# --------------------------------------------------
IN_CHANNELS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42
VAL_FRAC = 0.20
BASE_THRESHOLD = 0.50
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
POSTFIX_EVERY = 20
PRELOAD_SUBJECTS = True

# Remove connected components smaller than this many voxels.
# Conservative default so you do not accidentally remove real tiny lesions.
MIN_COMPONENT_VOXELS = 16

# How many worst subjects to save in the dedicated report
NUM_WORST_SUBJECTS = 15


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


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * dice_loss(logits, targets)


def resolve_checkpoint_path() -> Path:
    if BEST_CKPT_PATH.exists():
        return BEST_CKPT_PATH
    if LATEST_CKPT_PATH.exists():
        return LATEST_CKPT_PATH
    raise FileNotFoundError(
        f"Could not find a checkpoint.\n"
        f"Checked:\n"
        f"  {BEST_CKPT_PATH}\n"
        f"  {LATEST_CKPT_PATH}"
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def get_val_subjects(df: pd.DataFrame):
    all_subjects = sorted(df["subject_id"].unique().tolist())

    if VAL_SUBJECTS_TXT.exists():
        val_subjects = [
            line.strip()
            for line in VAL_SUBJECTS_TXT.read_text().splitlines()
            if line.strip()
        ]
        missing = sorted(set(val_subjects) - set(all_subjects))
        if missing:
            raise RuntimeError(
                "Some subjects listed in val_subjects.txt were not found in metadata:\n"
                + "\n".join(missing[:20])
            )
        return val_subjects, "loaded from logs/val_subjects.txt"

    subjects = all_subjects.copy()
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * VAL_FRAC))
    return subjects[:n_val], "reconstructed from RANDOM_SEED/VAL_FRAC fallback"


def binary_dice_np(pred_mask: np.ndarray, true_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    inter = np.logical_and(pred_mask, true_mask).sum(dtype=np.int64)
    denom = pred_mask.sum(dtype=np.int64) + true_mask.sum(dtype=np.int64)
    return float((2.0 * inter + eps) / (denom + eps))


def slice_dice_np(pred_vol: np.ndarray, true_vol: np.ndarray, eps=1e-6) -> np.ndarray:
    pred_vol = np.asarray(pred_vol, dtype=bool)
    true_vol = np.asarray(true_vol, dtype=bool)

    inter = np.logical_and(pred_vol, true_vol).sum(axis=(1, 2), dtype=np.int64)
    denom = pred_vol.sum(axis=(1, 2), dtype=np.int64) + true_vol.sum(axis=(1, 2), dtype=np.int64)
    return (2.0 * inter + eps) / (denom + eps)


def remove_small_components_3d(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)

    if min_voxels <= 1 or mask.sum() == 0:
        return mask.copy()

    if HAVE_SCIPY:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, num = ndi.label(mask, structure=structure)
        if num == 0:
            return mask.copy()

        counts = np.bincount(labeled.ravel())
        remove = counts < min_voxels
        remove[0] = False

        cleaned = mask.copy()
        cleaned[remove[labeled]] = False
        return cleaned

    # Fallback if scipy is unavailable:
    # remove small components slice-by-slice in 2D.
    cleaned = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        sl = mask[z].astype(np.uint8)
        if sl.sum() == 0:
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sl, connectivity=8)
        keep = np.zeros_like(sl, dtype=bool)

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area >= min_voxels:
                keep[labels == label_idx] = True

        cleaned[z] = keep

    return cleaned


def assign_size_groups_by_area(df: pd.DataFrame, area_col: str = "lesion_area", group_col: str = "size_group"):
    df = df.copy()

    if area_col not in df.columns or df.empty:
        df[group_col] = pd.Series(dtype=object)
        return df

    sorted_idx = df[area_col].sort_values(ascending=True).index.tolist()
    n = len(sorted_idx)

    groups = pd.Series(index=df.index, dtype=object)

    if n < 3:
        for idx in sorted_idx:
            groups.loc[idx] = "small"
    else:
        small_end = n // 3
        medium_end = 2 * n // 3

        for rank, idx in enumerate(sorted_idx):
            if rank < small_end:
                groups.loc[idx] = "small"
            elif rank < medium_end:
                groups.loc[idx] = "medium"
            else:
                groups.loc[idx] = "large"

    df[group_col] = groups
    return df


def main():
    set_seed(RANDOM_SEED)

    thresholds = sorted(set(float(t) for t in THRESHOLDS + [BASE_THRESHOLD]))

    if not META_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {META_CSV}\n"
            "Run build_runtime_metadata_flair_t1gd_25d5.py first."
        )

    df = pd.read_csv(META_CSV)
    if df.empty:
        raise RuntimeError("metadata.csv is empty.")

    val_subjects, split_source = get_val_subjects(df)

    val_df = (
        df[df["subject_id"].isin(val_subjects)]
        .copy()
        .sort_values(["subject_id", "slice_idx"])
        .reset_index(drop=True)
    )

    if val_df.empty:
        raise RuntimeError("Validation dataframe is empty after subject filtering.")

    ckpt_path = resolve_checkpoint_path()

    print("Experiment:", EXPERIMENT_NAME)
    print("Data experiment:", DATA_EXPERIMENT_NAME)
    print("Metadata:", META_CSV)
    print("Checkpoint:", ckpt_path)
    print("Validation subjects source:", split_source)
    print("Val subjects:", len(val_subjects))
    print("Val slices:", len(val_df))
    print("Val positives:", int(val_df["has_lesion"].sum()))
    print("Base threshold:", BASE_THRESHOLD)
    print("Threshold sweep:", thresholds)
    print("Preload subjects:", PRELOAD_SUBJECTS)
    print("Min component voxels:", MIN_COMPONENT_VOXELS)
    print("Component mode:", "3D scipy" if HAVE_SCIPY else "2D cv2 fallback")

    val_ds = BrainTumorSliceDataset(
        val_df,
        augment=False,
        preload_subjects=PRELOAD_SUBJECTS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = build_model().to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    threshold_stats = {
        thr: {
            "slice_seen": 0,
            "pos_slice_seen": 0,
            "slice_dice_sum_raw": 0.0,
            "slice_dice_sum_pp": 0.0,
            "pos_slice_dice_sum_raw": 0.0,
            "pos_slice_dice_sum_pp": 0.0,
            "global_inter_raw": 0.0,
            "global_pred_sum_raw": 0.0,
            "global_inter_pp": 0.0,
            "global_pred_sum_pp": 0.0,
            "global_target_sum": 0.0,
            "subject_dices_raw": [],
            "subject_dices_pp": [],
        }
        for thr in thresholds
    }

    val_loss_sum = 0.0
    val_seen = 0

    slice_rows = []
    subject_rows = []

    current_subject_id = None
    current_prob_slices = []
    current_target_slices = []
    current_meta_rows = []

    sample_offset = 0

    def finalize_current_subject():
        nonlocal current_subject_id
        nonlocal current_prob_slices
        nonlocal current_target_slices
        nonlocal current_meta_rows
        nonlocal slice_rows
        nonlocal subject_rows

        if current_subject_id is None:
            return

        probs_vol = np.stack(current_prob_slices, axis=0).astype(np.float32, copy=False)   # [D,H,W]
        target_vol = np.stack(current_target_slices, axis=0).astype(bool, copy=False)       # [D,H,W]

        num_slices = int(probs_vol.shape[0])
        target_pixels_total = int(target_vol.sum())
        pos_slice_mask = target_vol.reshape(num_slices, -1).sum(axis=1) > 0
        num_positive_slices = int(pos_slice_mask.sum())

        base_pred_raw = None
        base_pred_pp = None
        base_slice_dice_raw = None
        base_slice_dice_pp = None
        base_subject_dice_raw = None
        base_subject_dice_pp = None

        for thr in thresholds:
            pred_raw = probs_vol > thr
            pred_pp = remove_small_components_3d(pred_raw, MIN_COMPONENT_VOXELS)

            slice_d_raw = slice_dice_np(pred_raw, target_vol)
            slice_d_pp = slice_dice_np(pred_pp, target_vol)

            subject_dice_raw = binary_dice_np(pred_raw, target_vol)
            subject_dice_pp = binary_dice_np(pred_pp, target_vol)

            inter_raw = np.logical_and(pred_raw, target_vol).sum(dtype=np.int64)
            pred_sum_raw = pred_raw.sum(dtype=np.int64)

            inter_pp = np.logical_and(pred_pp, target_vol).sum(dtype=np.int64)
            pred_sum_pp = pred_pp.sum(dtype=np.int64)

            stats = threshold_stats[thr]
            stats["slice_seen"] += num_slices
            stats["pos_slice_seen"] += num_positive_slices

            stats["slice_dice_sum_raw"] += float(slice_d_raw.sum())
            stats["slice_dice_sum_pp"] += float(slice_d_pp.sum())

            if num_positive_slices > 0:
                stats["pos_slice_dice_sum_raw"] += float(slice_d_raw[pos_slice_mask].sum())
                stats["pos_slice_dice_sum_pp"] += float(slice_d_pp[pos_slice_mask].sum())

            stats["global_inter_raw"] += float(inter_raw)
            stats["global_pred_sum_raw"] += float(pred_sum_raw)
            stats["global_inter_pp"] += float(inter_pp)
            stats["global_pred_sum_pp"] += float(pred_sum_pp)
            stats["global_target_sum"] += float(target_pixels_total)

            stats["subject_dices_raw"].append(float(subject_dice_raw))
            stats["subject_dices_pp"].append(float(subject_dice_pp))

            if abs(thr - BASE_THRESHOLD) < 1e-12:
                base_pred_raw = pred_raw
                base_pred_pp = pred_pp
                base_slice_dice_raw = slice_d_raw
                base_slice_dice_pp = slice_d_pp
                base_subject_dice_raw = float(subject_dice_raw)
                base_subject_dice_pp = float(subject_dice_pp)

        if base_pred_raw is None:
            raise RuntimeError("BASE_THRESHOLD was not found in the threshold sweep.")

        subject_rows.append({
            "subject_id": current_subject_id,
            "num_slices": num_slices,
            "num_positive_slices": num_positive_slices,
            "target_pixels": target_pixels_total,
            "pred_pixels_raw": int(base_pred_raw.sum()),
            "pred_pixels_pp": int(base_pred_pp.sum()),
            "raw_subject_dice": base_subject_dice_raw,
            "pp_subject_dice": base_subject_dice_pp,
        })

        for i, meta in enumerate(current_meta_rows):
            row = {
                "subject_id": current_subject_id,
                "slice_idx": int(meta["slice_idx"]),
                "has_lesion": int(meta["has_lesion"]),
                "target_pixels": int(target_vol[i].sum()),
                "pred_pixels_raw": int(base_pred_raw[i].sum()),
                "pred_pixels_pp": int(base_pred_pp[i].sum()),
                "raw_dice": float(base_slice_dice_raw[i]),
                "pp_dice": float(base_slice_dice_pp[i]),
                "max_prob": float(probs_vol[i].max()),
                "mean_prob": float(probs_vol[i].mean()),
            }

            if meta["lesion_area"] is not None and not pd.isna(meta["lesion_area"]):
                row["lesion_area"] = float(meta["lesion_area"])

            slice_rows.append(row)

        current_subject_id = None
        current_prob_slices = []
        current_target_slices = []
        current_meta_rows = []

    val_pbar = tqdm(
        val_loader,
        desc="Evaluating",
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

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
            y_np = (y.squeeze(1).cpu().numpy() > 0.5)

            bs = x.size(0)
            val_loss_sum += loss.item() * bs
            val_seen += bs

            batch_meta = val_df.iloc[sample_offset: sample_offset + bs]

            for i, row in enumerate(batch_meta.itertuples(index=False)):
                subject_id = row.subject_id

                if current_subject_id is None:
                    current_subject_id = subject_id

                if subject_id != current_subject_id:
                    finalize_current_subject()
                    current_subject_id = subject_id

                current_prob_slices.append(probs[i])
                current_target_slices.append(y_np[i])
                current_meta_rows.append({
                    "slice_idx": int(row.slice_idx),
                    "has_lesion": int(row.has_lesion),
                    "lesion_area": getattr(row, "lesion_area", None),
                })

            sample_offset += bs

            if batch_idx % POSTFIX_EVERY == 0 or batch_idx == len(val_loader):
                base_stats = threshold_stats[BASE_THRESHOLD]
                raw_dice_now = base_stats["slice_dice_sum_raw"] / max(1, base_stats["slice_seen"])
                raw_pos_dice_now = base_stats["pos_slice_dice_sum_raw"] / max(1, base_stats["pos_slice_seen"])
                pp_dice_now = base_stats["slice_dice_sum_pp"] / max(1, base_stats["slice_seen"])
                val_pbar.set_postfix({
                    "loss": f"{val_loss_sum / max(1, val_seen):.4f}",
                    "raw_dice": f"{raw_dice_now:.4f}",
                    "raw_pos": f"{raw_pos_dice_now:.4f}",
                    "pp_dice": f"{pp_dice_now:.4f}",
                })

    finalize_current_subject()

    val_loss = val_loss_sum / max(1, val_seen)

    threshold_rows = []
    for thr in thresholds:
        stats = threshold_stats[thr]

        raw_val_dice_mean = stats["slice_dice_sum_raw"] / max(1, stats["slice_seen"])
        pp_val_dice_mean = stats["slice_dice_sum_pp"] / max(1, stats["slice_seen"])

        raw_val_pos_dice_mean = stats["pos_slice_dice_sum_raw"] / max(1, stats["pos_slice_seen"])
        pp_val_pos_dice_mean = stats["pos_slice_dice_sum_pp"] / max(1, stats["pos_slice_seen"])

        raw_global_pixel_dice = (
            (2.0 * stats["global_inter_raw"] + 1e-6)
            / (stats["global_pred_sum_raw"] + stats["global_target_sum"] + 1e-6)
        )
        pp_global_pixel_dice = (
            (2.0 * stats["global_inter_pp"] + 1e-6)
            / (stats["global_pred_sum_pp"] + stats["global_target_sum"] + 1e-6)
        )

        threshold_rows.append({
            "threshold": float(thr),

            "raw_val_dice_mean": float(raw_val_dice_mean),
            "raw_val_pos_dice_mean": float(raw_val_pos_dice_mean),
            "raw_global_pixel_dice": float(raw_global_pixel_dice),
            "raw_subject_dice_mean": float(np.mean(stats["subject_dices_raw"])) if stats["subject_dices_raw"] else float("nan"),
            "raw_subject_dice_median": float(np.median(stats["subject_dices_raw"])) if stats["subject_dices_raw"] else float("nan"),

            "pp_val_dice_mean": float(pp_val_dice_mean),
            "pp_val_pos_dice_mean": float(pp_val_pos_dice_mean),
            "pp_global_pixel_dice": float(pp_global_pixel_dice),
            "pp_subject_dice_mean": float(np.mean(stats["subject_dices_pp"])) if stats["subject_dices_pp"] else float("nan"),
            "pp_subject_dice_median": float(np.median(stats["subject_dices_pp"])) if stats["subject_dices_pp"] else float("nan"),
        })

    threshold_metrics_df = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)

    base_row = threshold_metrics_df.loc[np.isclose(threshold_metrics_df["threshold"], BASE_THRESHOLD)]
    if base_row.empty:
        raise RuntimeError("Could not find BASE_THRESHOLD row in threshold_metrics_df.")
    base_row = base_row.iloc[0]

    best_raw_row = threshold_metrics_df.iloc[threshold_metrics_df["raw_val_pos_dice_mean"].idxmax()]
    best_pp_row = threshold_metrics_df.iloc[threshold_metrics_df["pp_val_pos_dice_mean"].idxmax()]

    slice_metrics_df = pd.DataFrame(slice_rows).sort_values(["subject_id", "slice_idx"]).reset_index(drop=True)
    subject_metrics_df = pd.DataFrame(subject_rows).sort_values("subject_id").reset_index(drop=True)

    size_group_metrics_df = pd.DataFrame()
    if "lesion_area" in slice_metrics_df.columns:
        pos_slice_df = slice_metrics_df[slice_metrics_df["has_lesion"] == 1].copy()
        if not pos_slice_df.empty:
            pos_slice_df = assign_size_groups_by_area(pos_slice_df, area_col="lesion_area", group_col="size_group")

            size_group_metrics_df = (
                pos_slice_df
                .groupby("size_group", sort=False)
                .agg(
                    n_slices=("slice_idx", "count"),
                    lesion_area_mean=("lesion_area", "mean"),
                    lesion_area_median=("lesion_area", "median"),
                    raw_dice_mean=("raw_dice", "mean"),
                    raw_dice_median=("raw_dice", "median"),
                    pp_dice_mean=("pp_dice", "mean"),
                    pp_dice_median=("pp_dice", "median"),
                )
                .reset_index()
            )

            order = {"small": 0, "medium": 1, "large": 2}
            size_group_metrics_df["sort_order"] = size_group_metrics_df["size_group"].map(order)
            size_group_metrics_df = (
                size_group_metrics_df
                .sort_values("sort_order")
                .drop(columns=["sort_order"])
                .reset_index(drop=True)
            )

    worst_subjects_df = (
        subject_metrics_df
        .sort_values(["raw_subject_dice", "pp_subject_dice", "target_pixels"], ascending=[True, True, False])
        .head(NUM_WORST_SUBJECTS)
        .reset_index(drop=True)
    )

    neg_slice_df = slice_metrics_df[slice_metrics_df["has_lesion"] == 0].copy()
    pos_slice_df = slice_metrics_df[slice_metrics_df["has_lesion"] == 1].copy()

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "data_experiment_name": DATA_EXPERIMENT_NAME,
        "checkpoint_path": str(ckpt_path),
        "metadata_path": str(META_CSV),
        "val_subjects_source": split_source,
        "num_val_subjects": int(len(val_subjects)),
        "num_val_slices": int(len(val_df)),
        "num_val_positive_slices": int(val_df["has_lesion"].sum()),

        "batch_size": int(BATCH_SIZE),
        "num_workers": int(NUM_WORKERS),
        "preload_subjects": bool(PRELOAD_SUBJECTS),

        "base_threshold": float(BASE_THRESHOLD),
        "min_component_voxels": int(MIN_COMPONENT_VOXELS),
        "component_mode": "3D scipy" if HAVE_SCIPY else "2D cv2 fallback",

        "val_loss": float(val_loss),

        "raw_val_dice_mean": float(base_row["raw_val_dice_mean"]),
        "raw_val_pos_dice_mean": float(base_row["raw_val_pos_dice_mean"]),
        "raw_global_pixel_dice": float(base_row["raw_global_pixel_dice"]),
        "raw_subject_dice_mean": float(base_row["raw_subject_dice_mean"]),
        "raw_subject_dice_median": float(base_row["raw_subject_dice_median"]),

        "pp_val_dice_mean": float(base_row["pp_val_dice_mean"]),
        "pp_val_pos_dice_mean": float(base_row["pp_val_pos_dice_mean"]),
        "pp_global_pixel_dice": float(base_row["pp_global_pixel_dice"]),
        "pp_subject_dice_mean": float(base_row["pp_subject_dice_mean"]),
        "pp_subject_dice_median": float(base_row["pp_subject_dice_median"]),

        "negative_slice_with_pred_frac_raw": float((neg_slice_df["pred_pixels_raw"] > 0).mean()) if not neg_slice_df.empty else float("nan"),
        "negative_slice_with_pred_frac_pp": float((neg_slice_df["pred_pixels_pp"] > 0).mean()) if not neg_slice_df.empty else float("nan"),
        "positive_slice_missed_frac_raw": float((pos_slice_df["pred_pixels_raw"] == 0).mean()) if not pos_slice_df.empty else float("nan"),
        "positive_slice_missed_frac_pp": float((pos_slice_df["pred_pixels_pp"] == 0).mean()) if not pos_slice_df.empty else float("nan"),

        "best_raw_threshold_by_val_pos_dice": float(best_raw_row["threshold"]),
        "best_raw_val_pos_dice_mean": float(best_raw_row["raw_val_pos_dice_mean"]),
        "best_pp_threshold_by_val_pos_dice": float(best_pp_row["threshold"]),
        "best_pp_val_pos_dice_mean": float(best_pp_row["pp_val_pos_dice_mean"]),
    }

    summary_df = pd.DataFrame([summary])

    summary_path = EVAL_DIR / "evaluation_summary.csv"
    threshold_sweep_path = EVAL_DIR / "threshold_sweep.csv"
    slice_metrics_path = EVAL_DIR / "slice_metrics.csv"
    subject_metrics_path = EVAL_DIR / "subject_metrics.csv"
    size_group_metrics_path = EVAL_DIR / "size_group_metrics.csv"
    worst_subjects_path = EVAL_DIR / "worst_subjects.csv"

    summary_df.to_csv(summary_path, index=False)
    threshold_metrics_df.to_csv(threshold_sweep_path, index=False)
    slice_metrics_df.to_csv(slice_metrics_path, index=False)
    subject_metrics_df.to_csv(subject_metrics_path, index=False)

    if not size_group_metrics_df.empty:
        size_group_metrics_df.to_csv(size_group_metrics_path, index=False)

    worst_subjects_df.to_csv(worst_subjects_path, index=False)

    print("\nEvaluation complete.")
    print(f"val_loss                = {val_loss:.4f}")
    print(f"raw_val_dice            = {base_row['raw_val_dice_mean']:.4f}")
    print(f"raw_val_pos_dice        = {base_row['raw_val_pos_dice_mean']:.4f}")
    print(f"raw_global_pixel_dice   = {base_row['raw_global_pixel_dice']:.4f}")
    print(f"raw_subject_dice_mean   = {base_row['raw_subject_dice_mean']:.4f}")
    print(f"raw_subject_dice_median = {base_row['raw_subject_dice_median']:.4f}")

    print(f"pp_val_dice             = {base_row['pp_val_dice_mean']:.4f}")
    print(f"pp_val_pos_dice         = {base_row['pp_val_pos_dice_mean']:.4f}")
    print(f"pp_global_pixel_dice    = {base_row['pp_global_pixel_dice']:.4f}")
    print(f"pp_subject_dice_mean    = {base_row['pp_subject_dice_mean']:.4f}")
    print(f"pp_subject_dice_median  = {base_row['pp_subject_dice_median']:.4f}")

    print(f"\nBest raw threshold by val_pos_dice: {best_raw_row['threshold']:.2f} "
          f"(val_pos_dice={best_raw_row['raw_val_pos_dice_mean']:.4f})")
    print(f"Best pp threshold by val_pos_dice:  {best_pp_row['threshold']:.2f} "
          f"(val_pos_dice={best_pp_row['pp_val_pos_dice_mean']:.4f})")

    if not size_group_metrics_df.empty:
        print("\nPositive-slice size breakdown:")
        print(size_group_metrics_df.to_string(index=False))

    if not worst_subjects_df.empty:
        print("\nWorst subjects (by raw subject dice):")
        print(
            worst_subjects_df[
                [
                    "subject_id",
                    "num_positive_slices",
                    "target_pixels",
                    "pred_pixels_raw",
                    "pred_pixels_pp",
                    "raw_subject_dice",
                    "pp_subject_dice",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved summary to:        {summary_path}")
    print(f"Saved threshold sweep to:{threshold_sweep_path}")
    print(f"Saved slice metrics to:  {slice_metrics_path}")
    print(f"Saved subject metrics to:{subject_metrics_path}")
    if not size_group_metrics_df.empty:
        print(f"Saved size breakdown to: {size_group_metrics_path}")
    print(f"Saved worst subjects to: {worst_subjects_path}")


if __name__ == "__main__":
    main()