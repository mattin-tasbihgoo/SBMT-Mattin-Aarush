from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images"
MSK_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

USE_CHANNEL = 0  # 0=FLAIR, 1=T1w, 2=t1gd, 3=T2w

IMG_DIR.mkdir(parents=True, exist_ok=True)
MSK_DIR.mkdir(parents=True, exist_ok=True)

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

    for img_path in tqdm(image_files, desc="Preprocessing"):
        subject_id = img_path.name.replace(".nii.gz", "")
        lbl_path = label_dir / f"{subject_id}.nii.gz"

        if not lbl_path.exists():
            print(f"Missing label for {subject_id}")
            continue

        img = nib.load(str(img_path)).get_fdata()   # (240,240,155,4)
        lbl = nib.load(str(lbl_path)).get_fdata()   # (240,240,155)

        img = img[:, :, :, USE_CHANNEL]
        lbl = (lbl > 0).astype(np.uint8)  # binary whole-tumor mask

        img = normalize_volume(img)

        z_start = int(img.shape[2] * 0.20)
        z_end = int(img.shape[2] * 0.90)

        for z in range(z_start, z_end):
            img2d = img[:, :, z]
            msk2d = lbl[:, :, z]

            brain_pixels = int(np.count_nonzero(img2d))
            has_lesion = int(msk2d.sum() > 0)

            if brain_pixels < 500 and has_lesion == 0:
                continue

            img2d = resize_image(img2d)
            msk2d = resize_mask(msk2d)

            img_name = f"{subject_id}_z{z:03d}.npy"
            msk_name = f"{subject_id}_z{z:03d}.npy"

            np.save(IMG_DIR / img_name, img2d.astype(np.float32))
            np.save(MSK_DIR / msk_name, msk2d.astype(np.uint8))

            records.append({
                "subject_id": subject_id,
                "slice_idx": z,
                "image_path": str(IMG_DIR / img_name),
                "mask_path": str(MSK_DIR / msk_name),
                "has_lesion": has_lesion
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