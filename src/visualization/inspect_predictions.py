import sys
from pathlib import Path
import io
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.networks.nets import UNet
from PIL import Image, ImageDraw, ImageFont
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import FLAIR_CH, T1GD_CH, normalize_volume, resize_image, resize_mask

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    ndi = None
    HAVE_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_EXPERIMENT_NAME = "runtime_flair_t1gd_25d5"

# Folder containing the frozen or current run artifacts.
RUN_DIR_NAME = "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"

# If the checkpoint filename stem differs from the folder name,
# keep it here. This matches your frozen baseline setup.
CHECKPOINT_EXPERIMENT_NAME = "runtime_flair_t1gd_25d5_smalllesion"

DATA_ROOT = PROJECT_ROOT / "data" / DATA_EXPERIMENT_NAME
META_CSV = DATA_ROOT / "metadata.csv"

OUT_ROOT = PROJECT_ROOT / "outputs" / RUN_DIR_NAME
CKPT_DIR = OUT_ROOT / "checkpoints"
CKPT_PATH = CKPT_DIR / f"best_model_{CHECKPOINT_EXPERIMENT_NAME}.pt"
LOG_DIR = OUT_ROOT / "logs"
EVAL_DIR = OUT_ROOT / "evaluation"
INSPECT_DIR = OUT_ROOT / "inspection"

VAL_SUBJECTS_TXT = LOG_DIR / "val_subjects.txt"
WORST_SUBJECTS_CSV = EVAL_DIR / "worst_subjects.csv"
SUBJECT_METRICS_CSV = EVAL_DIR / "subject_metrics.csv"

INSPECT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Frozen baseline inference settings
# --------------------------------------------------
IN_CHANNELS = 10
OUT_SIZE = (256, 256)
OFFSETS = (-2, -1, 0, 1, 2)
THRESHOLD = 0.60
MIN_COMPONENT_VOXELS = 16
INFER_BATCH_SIZE = 16
RANDOM_SEED = 42

# Which subjects to inspect
MANUAL_SUBJECT_IDS = ["BRATS_381", "BRATS_101", "BRATS_376"]
ADD_TOP_BEST_SUBJECTS = 2

# Slice export behavior
SAVE_ONLY_INTERESTING_SLICES = True
MAX_SLICES_PER_SUBJECT = 16

# PDF creation
CREATE_PER_SUBJECT_PDF = True
CREATE_MASTER_PDF = True
MASTER_PDF_NAME = "inspection_selected_subjects.pdf"


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


def resolve_checkpoint_path() -> Path:
    if CKPT_PATH.exists():
        return CKPT_PATH

    candidates = sorted(CKPT_DIR.glob("best_model_*.pt"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"Missing checkpoint in: {CKPT_DIR}\n"
            f"Expected: {CKPT_PATH}"
        )

    raise RuntimeError(
        "Multiple best checkpoints were found. Set CHECKPOINT_EXPERIMENT_NAME explicitly.\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: str):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
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


def clamp_idx(z: int, depth: int) -> int:
    return max(0, min(z, depth - 1))


def binary_dice_np(pred_mask: np.ndarray, true_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    inter = np.logical_and(pred_mask, true_mask).sum(dtype=np.int64)
    denom = pred_mask.sum(dtype=np.int64) + true_mask.sum(dtype=np.int64)
    return float((2.0 * inter + eps) / (denom + eps))


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

    return mask.copy()


def get_val_subjects(df: pd.DataFrame):
    if not VAL_SUBJECTS_TXT.exists():
        raise FileNotFoundError(f"Missing validation subject file: {VAL_SUBJECTS_TXT}")

    val_subjects = [
        line.strip()
        for line in VAL_SUBJECTS_TXT.read_text().splitlines()
        if line.strip()
    ]

    available = set(df["subject_id"].unique().tolist())
    val_subjects = [s for s in val_subjects if s in available]
    return val_subjects


def choose_subject_ids() -> list[str]:
    chosen = []

    for s in MANUAL_SUBJECT_IDS:
        if s not in chosen:
            chosen.append(s)

    if ADD_TOP_BEST_SUBJECTS > 0 and SUBJECT_METRICS_CSV.exists():
        best_df = pd.read_csv(SUBJECT_METRICS_CSV)

        if "pp_subject_dice" in best_df.columns:
            best_col = "pp_subject_dice"
        elif "raw_subject_dice" in best_df.columns:
            best_col = "raw_subject_dice"
        elif "subject_dice" in best_df.columns:
            best_col = "subject_dice"
        else:
            best_col = None

        if best_col is not None:
            best_ids = (
                best_df.sort_values(best_col, ascending=False)["subject_id"]
                .head(ADD_TOP_BEST_SUBJECTS)
                .tolist()
            )
            for s in best_ids:
                if s not in chosen:
                    chosen.append(s)

    if not chosen and WORST_SUBJECTS_CSV.exists():
        worst_df = pd.read_csv(WORST_SUBJECTS_CSV)
        chosen = worst_df["subject_id"].head(3).tolist()

    return chosen


def load_subject_volumes(image_path: Path, label_path: Path):
    img4d = nib.load(str(image_path)).get_fdata().astype(np.float32)
    lbl = nib.load(str(label_path)).get_fdata().astype(np.float32)

    flair = normalize_volume(img4d[:, :, :, FLAIR_CH])
    t1gd = normalize_volume(img4d[:, :, :, T1GD_CH])
    lbl = (lbl > 0).astype(np.uint8)

    # [H, W, D] -> [D, H, W]
    flair = np.ascontiguousarray(np.transpose(flair, (2, 0, 1)), dtype=np.float32)
    t1gd = np.ascontiguousarray(np.transpose(t1gd, (2, 0, 1)), dtype=np.float32)
    lbl = np.ascontiguousarray(np.transpose(lbl, (2, 0, 1)), dtype=np.uint8)

    return flair, t1gd, lbl


def make_input_for_slice(flair: np.ndarray, t1gd: np.ndarray, z: int):
    depth = flair.shape[0]
    idxs = [clamp_idx(z + dz, depth) for dz in OFFSETS]

    flair_slices = [resize_image(flair[i], OUT_SIZE) for i in idxs]
    t1gd_slices = [resize_image(t1gd[i], OUT_SIZE) for i in idxs]

    x = np.stack(flair_slices + t1gd_slices, axis=0).astype(np.float32)
    return x


def to_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img


def make_overlay(base_img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = to_display(base_img)
    rgb = np.stack([base, base, base], axis=-1)

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    rgb[gt & ~pred] = [0.0, 1.0, 0.0]   # green = missed GT
    rgb[pred & ~gt] = [1.0, 0.0, 0.0]   # red = false positive
    rgb[gt & pred] = [1.0, 1.0, 0.0]    # yellow = overlap

    return rgb


def choose_slices_to_save(gt_small: np.ndarray, pred_pp: np.ndarray, probs: np.ndarray):
    depth = gt_small.shape[0]

    if SAVE_ONLY_INTERESTING_SLICES:
        idxs = [z for z in range(depth) if gt_small[z].sum() > 0 or pred_pp[z].sum() > 0]
        if not idxs:
            idxs = list(range(depth))
    else:
        idxs = list(range(depth))

    if len(idxs) <= MAX_SLICES_PER_SUBJECT:
        return idxs

    scored = []
    for z in idxs:
        score = (
            3.0 * float(gt_small[z].sum())
            + 1.0 * float(pred_pp[z].sum())
            + 500.0 * float(probs[z].max())
        )
        scored.append((score, z))

    chosen = [z for _, z in sorted(scored, reverse=True)[:MAX_SLICES_PER_SUBJECT]]
    return sorted(chosen)


def render_slice_figure(
    subject_id: str,
    z: int,
    flair_img: np.ndarray,
    t1gd_img: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
) -> Image.Image:
    overlay = make_overlay(flair_img, gt, pred)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    axes[0].imshow(flair_img, cmap="gray")
    axes[0].set_title("FLAIR")
    axes[0].axis("off")

    axes[1].imshow(t1gd_img, cmap="gray")
    axes[1].set_title("T1-Gd")
    axes[1].axis("off")

    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    axes[5].imshow(prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[5].set_title("Probability")
    axes[5].axis("off")

    gt_px = int(gt.sum())
    pred_px = int(pred.sum())
    max_prob = float(prob.max())

    fig.suptitle(
        f"{subject_id} | z={z} | gt_px={gt_px} | pred_px={pred_px} | max_prob={max_prob:.3f}",
        fontsize=11,
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_pil_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def make_text_page(title: str, subtitle_lines: list[str], size=(1654, 2339)) -> Image.Image:
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()

    y = 120
    draw.text((100, y), title, fill="black", font=font)
    y += 60

    for line in subtitle_lines:
        draw.text((100, y), line, fill="black", font=font)
        y += 30

    return page


def save_images_as_pdf(images: list[Image.Image], out_pdf: Path):
    if not images:
        return

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    pdf_images = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        pdf_images.append(img)

    first, rest = pdf_images[0], pdf_images[1:]
    first.save(out_pdf, save_all=True, append_images=rest, resolution=150.0)


def inspect_subject(model, device: str, amp_device: str, subject_row: pd.Series):
    subject_id = subject_row["subject_id"]
    image_path = Path(subject_row["image_path"])
    label_path = Path(subject_row["label_path"])

    subject_dir = INSPECT_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    flair, t1gd, lbl = load_subject_volumes(image_path, label_path)
    depth = flair.shape[0]

    flair_small = np.stack([resize_image(flair[z], OUT_SIZE) for z in range(depth)], axis=0).astype(np.float32)
    t1gd_small = np.stack([resize_image(t1gd[z], OUT_SIZE) for z in range(depth)], axis=0).astype(np.float32)
    gt_small = np.stack([resize_mask(lbl[z], OUT_SIZE).astype(np.uint8) for z in range(depth)], axis=0)

    probs = np.zeros((depth, OUT_SIZE[1], OUT_SIZE[0]), dtype=np.float32)

    x_batch = []
    z_batch = []

    model.eval()
    with torch.no_grad():
        for z in tqdm(range(depth), desc=f"Infer {subject_id}", leave=False):
            x = make_input_for_slice(flair, t1gd, z)
            x_batch.append(x)
            z_batch.append(z)

            if len(x_batch) == INFER_BATCH_SIZE or z == depth - 1:
                xb = torch.from_numpy(np.stack(x_batch, axis=0)).to(device)

                with autocast(device_type=amp_device, enabled=(device == "cuda")):
                    logits = model(xb)
                    pb = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

                for local_i, zz in enumerate(z_batch):
                    probs[zz] = pb[local_i]

                x_batch = []
                z_batch = []

    pred_raw = probs > THRESHOLD
    pred_pp = remove_small_components_3d(pred_raw, MIN_COMPONENT_VOXELS)

    slice_ids = choose_slices_to_save(gt_small, pred_pp, probs)

    summary_rows = []
    pdf_pages = []

    title_page = make_text_page(
        title=f"Inspection Report: {subject_id}",
        subtitle_lines=[
            f"run_dir: {RUN_DIR_NAME}",
            f"checkpoint_file: {resolve_checkpoint_path().name}",
            f"threshold: {THRESHOLD}",
            f"min_component_voxels: {MIN_COMPONENT_VOXELS}",
            f"saved_slices: {len(slice_ids)}",
            f"scipy_3d_postprocess: {HAVE_SCIPY}",
        ],
    )
    pdf_pages.append(title_page)

    for z in slice_ids:
        fig_img = render_slice_figure(
            subject_id=subject_id,
            z=z,
            flair_img=flair_small[z],
            t1gd_img=t1gd_small[z],
            gt=gt_small[z],
            pred=pred_pp[z].astype(np.uint8),
            prob=probs[z],
        )

        out_path = subject_dir / f"z_{z:03d}.png"
        save_pil_image(fig_img, out_path)
        pdf_pages.append(fig_img)

        summary_rows.append({
            "subject_id": subject_id,
            "slice_idx": z,
            "target_pixels": int(gt_small[z].sum()),
            "pred_pixels_raw": int(pred_raw[z].sum()),
            "pred_pixels_pp": int(pred_pp[z].sum()),
            "slice_dice_pp": binary_dice_np(pred_pp[z], gt_small[z]),
            "max_prob": float(probs[z].max()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(subject_dir / "slice_summary.csv", index=False)

    subject_dice = binary_dice_np(pred_pp, gt_small)
    info_text = (
        f"subject_id: {subject_id}\n"
        f"image_path: {image_path}\n"
        f"label_path: {label_path}\n"
        f"threshold: {THRESHOLD}\n"
        f"min_component_voxels: {MIN_COMPONENT_VOXELS}\n"
        f"saved_slices: {len(slice_ids)}\n"
        f"subject_dice_pp: {subject_dice:.6f}\n"
    )
    (subject_dir / "subject_info.txt").write_text(info_text)

    per_subject_pdf_path = None
    if CREATE_PER_SUBJECT_PDF:
        per_subject_pdf_path = subject_dir / f"{subject_id}_inspection.pdf"
        save_images_as_pdf(pdf_pages, per_subject_pdf_path)

    print(f"Saved {len(slice_ids)} slice figures for {subject_id} -> {subject_dir}")

    return {
        "subject_id": subject_id,
        "subject_dir": subject_dir,
        "pdf_pages": pdf_pages,
        "per_subject_pdf_path": per_subject_pdf_path,
    }


def main():
    set_seed(RANDOM_SEED)

    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_CSV}")

    ckpt_path = resolve_checkpoint_path()

    df = pd.read_csv(META_CSV)
    val_subjects = set(get_val_subjects(df))

    subject_df = (
        df[df["subject_id"].isin(val_subjects)][["subject_id", "image_path", "label_path"]]
        .drop_duplicates()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    chosen_ids = choose_subject_ids()
    if not chosen_ids:
        raise RuntimeError("No subject IDs selected for inspection.")

    chosen_df = subject_df[subject_df["subject_id"].isin(chosen_ids)].copy()
    if chosen_df.empty:
        raise RuntimeError("Chosen subject IDs were not found in the validation split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"

    print("Checkpoint:", ckpt_path)
    print("Inspection output:", INSPECT_DIR)
    print("Subjects:", chosen_df["subject_id"].tolist())
    print("Threshold:", THRESHOLD)
    print("Min component voxels:", MIN_COMPONENT_VOXELS)

    model = build_model().to(device)
    load_checkpoint(model, ckpt_path, device)

    master_pages = []
    report_rows = []

    if CREATE_MASTER_PDF:
        master_pages.append(
            make_text_page(
                title="Inspection Report: Selected Subjects",
                subtitle_lines=[
                    f"run_dir: {RUN_DIR_NAME}",
                    f"checkpoint_file: {ckpt_path.name}",
                    f"threshold: {THRESHOLD}",
                    f"min_component_voxels: {MIN_COMPONENT_VOXELS}",
                    "yellow = overlap",
                    "green = missed ground truth",
                    "red = false positive",
                ],
            )
        )

    for row in chosen_df.itertuples(index=False):
        result = inspect_subject(model, device, amp_device, pd.Series(row._asdict()))
        report_rows.append({
            "subject_id": result["subject_id"],
            "subject_dir": str(result["subject_dir"]),
            "per_subject_pdf_path": str(result["per_subject_pdf_path"]) if result["per_subject_pdf_path"] else "",
        })

        if CREATE_MASTER_PDF:
            master_pages.extend(result["pdf_pages"])

    inspection_index_path = INSPECT_DIR / "inspection_index.csv"
    pd.DataFrame(report_rows).to_csv(inspection_index_path, index=False)

    if CREATE_MASTER_PDF and master_pages:
        master_pdf_path = INSPECT_DIR / MASTER_PDF_NAME
        save_images_as_pdf(master_pages, master_pdf_path)
        print(f"Saved combined PDF to: {master_pdf_path}")

    print(f"Saved inspection index to: {inspection_index_path}")
    print("\nDone.")
    print(f"Open this folder: {INSPECT_DIR}")


if __name__ == "__main__":
    main()
