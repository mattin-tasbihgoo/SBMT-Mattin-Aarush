from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_flair_t1gd_binary"
IMG_DIR = PROCESSED_ROOT / "images"
MSK_DIR = PROCESSED_ROOT / "masks"
META_CSV = PROCESSED_ROOT / "metadata.csv"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MSK_DIR.mkdir(parents=True, exist_ok=True)

FLAIR_CH = 0
T1GD_CH = 2


def normalize_volume(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    brain = img[img > 0]
    if brain.size == 0:
        return img

    lo, hi = np.percentile(brain, [1, 99])
    img = np.clip(img, lo, hi)

    brain = img[img > 0]
    img = (img - brain.mean()) / (brain.std() + 1e-8)
    return img.astype(np.float32)


def resize_image(x, size=(256, 256)):
    return cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(x, size=(256, 256)):
    return cv2.resize(x.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


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

    for img_path in tqdm(image_files, desc="Preprocessing FLAIR+t1gd"):
        subject_id = img_path.name.replace(".nii.gz", "")
        lbl_path = label_dir / f"{subject_id}.nii.gz"

        if not lbl_path.exists():
            print(f"Missing label for {subject_id}")
            continue

        img4d = nib.load(str(img_path)).get_fdata().astype(np.float32)   # (240,240,155,4)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.float32)     # (240,240,155)

        flair = img4d[:, :, :, FLAIR_CH]
        t1gd = img4d[:, :, :, T1GD_CH]
        lbl = (lbl > 0).astype(np.uint8)

        # Brain mask BEFORE normalization
        brain_mask = ((flair > 0) | (t1gd > 0)).astype(np.uint8)

        flair = normalize_volume(flair)
        t1gd = normalize_volume(t1gd)

        z_start = int(flair.shape[2] * 0.20)
        z_end = int(flair.shape[2] * 0.90)

        for z in range(z_start, z_end):
            flair2d = flair[:, :, z]
            t1gd2d = t1gd[:, :, z]
            msk2d = lbl[:, :, z]

            brain_pixels = int(brain_mask[:, :, z].sum())
            has_lesion = int(msk2d.sum() > 0)

            if brain_pixels < 500 and has_lesion == 0:
                continue

            flair2d = resize_image(flair2d)
            t1gd2d = resize_image(t1gd2d)
            msk2d = resize_mask(msk2d)

            img2d = np.stack([flair2d, t1gd2d], axis=0).astype(np.float32)  # [2, H, W]

            img_name = f"{subject_id}_z{z:03d}.npy"
            msk_name = f"{subject_id}_z{z:03d}.npy"

            np.save(IMG_DIR / img_name, img2d)
            np.save(MSK_DIR / msk_name, msk2d.astype(np.uint8))

            records.append({
                "subject_id": subject_id,
                "slice_idx": z,
                "image_path": str(IMG_DIR / img_name),
                "mask_path": str(MSK_DIR / msk_name),
                "has_lesion": has_lesion,
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