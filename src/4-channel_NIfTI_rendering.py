import nibabel as nib
import napari

img = nib.load("data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
arr = img.get_fdata()          # shape: (240, 240, 155, 4)  — full float64, no truncation

modalities = ["T1", "T1ce", "T2", "FLAIR"]
viewer = napari.Viewer()

for i, name in enumerate(modalities):
    viewer.add_image(arr[..., i], name=name, colormap="gray", visible=(i == 3))

# Optionally overlay the label:
seg = nib.load("E:/Brain Lesion Detection/data/raw/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz")
viewer.add_labels(seg.get_fdata().astype(int), name="seg")

napari.run()