from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"

OUT_ROOT = PROJECT_ROOT / "data" / "runtime_flair_t1gd_25d5"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
META_CSV = OUT_ROOT / "metadata.csv"

FLAIR_CH = 0
T1GD_CH = 2


def main():
    image_dir = RAW_ROOT / "imagesTr"
    label_dir = RAW_ROOT / "labelsTr"

    image_files = sorted(
        p for p in image_dir.glob("*.nii.gz")
        if not p.name.startswith("._")
    )

    print(f"RAW_ROOT: {RAW_ROOT}")
    print(f"Found {len(image_files)} training image files")

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No .nii.gz files found in {image_dir}. "
            "Check that Task01_BrainTumour is inside data/raw."
        )

    records = []

    for img_path in tqdm(image_files, desc="Building runtime metadata"):
        subject_id = img_path.name.replace(".nii.gz", "")
        lbl_path = label_dir / f"{subject_id}.nii.gz"

        if not lbl_path.exists():
            print(f"Missing label for {subject_id}")
            continue

        img4d = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.float32)

        flair = img4d[:, :, :, FLAIR_CH]
        t1gd = img4d[:, :, :, T1GD_CH]
        lbl = (lbl > 0).astype(np.uint8)

        brain_mask = ((flair > 0) | (t1gd > 0)).astype(np.uint8)

        z_start = int(flair.shape[2] * 0.20)
        z_end = int(flair.shape[2] * 0.90)

        # Need z-2 and z+2 for 5-slice context
        z_start = max(z_start, 2)
        z_end = min(z_end, flair.shape[2] - 2)

        for z in range(z_start, z_end):
            brain_pixels = int(brain_mask[:, :, z].sum())
            lesion_area = int(lbl[:, :, z].sum())
            has_lesion = int(lesion_area > 0)

            if brain_pixels < 500 and has_lesion == 0:
                continue

            records.append({
                "subject_id": subject_id,
                "slice_idx": z,
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "has_lesion": has_lesion,
                "lesion_area": lesion_area,
            })

    if not records:
        raise RuntimeError("No slices were produced. Check RAW_ROOT and file layout.")

    df = pd.DataFrame(records)
    df.to_csv(META_CSV, index=False)

    print("\nSaved metadata to:", META_CSV)
    print(df.head())
    print("\nClass counts:")
    print(df["has_lesion"].value_counts())
    print("\nUnique subjects:", df["subject_id"].nunique())
    print("Total slices:", len(df))


if __name__ == "__main__":
    main()