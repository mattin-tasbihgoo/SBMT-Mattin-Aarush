"""Preprocess MSD Task01_BrainTumour NIfTI volumes for 3D U-Net training.

Reads raw multi-channel NIfTI, extracts FLAIR (ch 0) and T1gd (ch 2),
applies per-channel percentile-clip + min-max normalization to ~[0, 1],
binarizes labels, and saves per-subject .npy volumes plus a metadata CSV.

Output layout
─────────────
  data/processed_3d/images/{subject_id}.npy   float32  (2, H, W, D)
  data/processed_3d/masks/{subject_id}.npy    uint8    (H, W, D)
  data/processed_3d/metadata.csv
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"

OUT_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = OUT_ROOT / "images"
MSK_DIR = OUT_ROOT / "masks"
META_CSV = OUT_ROOT / "metadata.csv"

FLAIR_CH = 0
T1GD_CH = 2


def normalize_channel(vol: np.ndarray) -> np.ndarray:
    """Percentile-clip + min-max normalization on foreground voxels.

    Differs from the 2.5D z-score in dataset.py — intentional for the 3D
    branch (see plan doc / poster methods).
    """
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    fg = vol[vol > 0]
    if fg.size == 0:
        return vol

    p1, p99 = np.percentile(fg, [1, 99])
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol.astype(np.float32)


def main():
    image_dir = RAW_ROOT / "imagesTr"
    label_dir = RAW_ROOT / "labelsTr"

    image_files = sorted(
        p for p in image_dir.glob("*.nii.gz")
        if not p.name.startswith("._")
    )

    print(f"RAW_ROOT:  {RAW_ROOT}")
    print(f"OUT_ROOT:  {OUT_ROOT}")
    print(f"Found {len(image_files)} image files in imagesTr/")

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No .nii.gz files found in {image_dir}. "
            "Check that Task01_BrainTumour is inside data/raw/."
        )

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MSK_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for img_path in tqdm(image_files, desc="Preprocessing 3D volumes"):
        subject_id = img_path.name.replace(".nii.gz", "")
        lbl_path = label_dir / f"{subject_id}.nii.gz"

        if not lbl_path.exists():
            print(f"  WARNING: missing label for {subject_id}, skipping")
            continue

        img4d = nib.load(str(img_path)).get_fdata()   # (H, W, D, 4)
        lbl3d = nib.load(str(lbl_path)).get_fdata()    # (H, W, D)

        flair = normalize_channel(img4d[..., FLAIR_CH])
        t1gd = normalize_channel(img4d[..., T1GD_CH])

        # (C, H, W, D) — no transpose; consistent with downstream patch code
        volume = np.stack([flair, t1gd], axis=0)

        mask = (lbl3d > 0).astype(np.uint8)
        tumor_voxels = int(mask.sum())

        img_out = IMG_DIR / f"{subject_id}.npy"
        msk_out = MSK_DIR / f"{subject_id}.npy"

        np.save(img_out, volume)
        np.save(msk_out, mask)

        spatial = mask.shape  # (H, W, D)
        records.append({
            "subject_id": subject_id,
            "image_path": str(img_out),
            "mask_path": str(msk_out),
            "tumor_voxels": tumor_voxels,
            "shape_h": spatial[0],
            "shape_w": spatial[1],
            "shape_d": spatial[2],
        })

    if not records:
        raise RuntimeError(
            "No subjects were processed. Check RAW_ROOT and file layout."
        )

    df = pd.DataFrame(records)
    df.to_csv(META_CSV, index=False)

    print(f"\nProcessed {len(df)} subjects")
    print(f"Metadata saved to: {META_CSV}")
    print(f"Example image shape: {np.load(df.iloc[0]['image_path']).shape}")
    print(f"Example mask  shape: {np.load(df.iloc[0]['mask_path']).shape}")
    print(f"\nTumor voxel stats:")
    print(df["tumor_voxels"].describe())


if __name__ == "__main__":
    main()
