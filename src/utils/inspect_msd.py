from pathlib import Path
import json
import nibabel as nib
import numpy as np

root = Path(r"C:\Users\Matti\Brain Lesion Detection\Task01_BrainTumour")

# Show dataset metadata
dataset_json = root / "dataset.json"
with open(dataset_json, "r") as f:
    meta = json.load(f)

print("\n=== DATASET.JSON ===")
for key in ["name", "description", "tensorImageSize", "modality", "labels", "numTraining", "numTest"]:
    if key in meta:
        print(f"{key}: {meta[key]}")

# Pick one example case
img_path = root / "imagesTr" / "BRATS_001.nii.gz"
lbl_path = root / "labelsTr" / "BRATS_001.nii.gz"

print("\n=== FILES ===")
print("Image:", img_path)
print("Label:", lbl_path)

img_nii = nib.load(str(img_path))
lbl_nii = nib.load(str(lbl_path))

img = img_nii.get_fdata()
lbl = lbl_nii.get_fdata()

print("\n=== IMAGE INFO ===")
print("Image shape:", img.shape)
print("Image dtype:", img.dtype)
print("Image min/max:", np.min(img), np.max(img))

print("\n=== LABEL INFO ===")
print("Label shape:", lbl.shape)
print("Label dtype:", lbl.dtype)
print("Unique label values:", np.unique(lbl))

print("\n=== AFFINE CHECK ===")
print("Image affine:\n", img_nii.affine)
print("Label affine:\n", lbl_nii.affine)

print("\n=== BASIC CHECKS ===")
print("Shape match:", img.shape[:3] == lbl.shape[:3] if img.ndim == 4 else img.shape == lbl.shape)

if img.ndim == 4:
    print("This image appears to be multi-channel / 4D.")
    print("Number of channels:", img.shape[3])
elif img.ndim == 3:
    print("This image appears to be a single-channel 3D volume.")
else:
    print("Unexpected image dimensionality.")