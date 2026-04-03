# Brain Lesion Detection — AI-Based Neuroimaging Segmentation Pipeline

> **Context:** This repository is the neuroimaging component of a broader research project on AI-based prediction of chemotherapy-induced cognitive impairment (CICI), presented at the **SBMT 2026 NeuroTech Annual Convention**, Los Angeles, April 16–19.

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Environment Setup](#4-environment-setup)
5. [Data Visualization](#5-data-visualization)
6. [Pipeline Overview](#6-pipeline-overview)
7. [2.5D Pipeline](#7-25d-pipeline)
8. [3D Pipeline](#8-3d-pipeline)
   - [Step 1: Preprocessing](#step-1-preprocessing)
   - [Step 2: Training](#step-2-training)
   - [Step 3: Evaluation](#step-3-evaluation)
   - [Step 4: Visualization](#step-4-visualization)
9. [Results](#9-results)
10. [Glossary of All Terms](#10-glossary-of-all-terms)
11. [Understanding the Training Graphs](#11-understanding-the-training-graphs)
12. [File Reference](#12-file-reference)

---

## 1. Research Context

### The Clinical Problem

Chemotherapy-induced cognitive impairment (CICI), commonly called "chemo brain," is a well-documented but poorly predicted side effect of cancer treatment. Patients experience memory loss, difficulty concentrating, and reduced processing speed — sometimes lasting years after treatment ends. Despite the clinical burden, no rigorous AI model exists that can predict, before treatment, *which* patients will develop CICI.

A literature search across 469 papers on chemotherapy and cognition found that only 75 involved AI or machine learning methods, and **none provided drug-specific risk prediction models**. That gap is the scientific contribution of this work: demonstrating the need and beginning to build toward the solution.

### Why Brain Tumor Data?

A labeled CICI neuroimaging dataset does not exist in accessible form. As a methodologically sound proxy, this project builds the **neuroimaging submodule** of the eventual CICI prediction pipeline on the **MSD Task01_BrainTumour** dataset — a public, well-validated collection of multi-channel MRI scans with voxel-level tumor labels.

The imaging pipeline developed here is honestly framed as a **proof-of-concept neuroimaging module**, not a CICI predictor. It demonstrates that volumetric MRI analysis at this quality is feasible and reproducible on consumer hardware, and serves as the imaging arm of a future system that will integrate tabular patient data (demographics, drug regimens, inflammatory markers) with neuroimaging features.

### Progression of the Pipeline

| Version | Architecture | Best Validation Metric |
|---------|-------------|----------------------|
| Baseline FLAIR-only 2D U-Net | 2D, single channel | Dice 0.6335 (per-slice) |
| FLAIR + T1gd dual-channel | 2D, 2 channels | Improvement over baseline |
| 2.5D context (5-slice stack) | 2D, 10 channels | Best 2.5D result |
| **3D U-Net (current)** | **3D volumetric** | **Mean 3D Dice 0.894 / Median 0.923** |

---

## 2. Repository Structure

```
Brain-Lesion-Detection/
├── src/
│   │
│   ├── 4-channel_NIfTI_rendering.py  ← napari viewer for raw 4-channel NIfTI + labels
│   │
│   ├── ── 3D Pipeline ──────────────────────────────────────────────
│   ├── preprocess_3d.py              ← NIfTI → normalized .npy volumes + metadata CSV
│   ├── dataset_3d.py                 ← patch-based PyTorch Dataset (foreground-biased)
│   ├── model_3d.py                   ← UNet3D architecture (Conv3d throughout)
│   ├── sliding_window.py             ← overlapping patch inference for full-volume output
│   ├── train_3d.py                   ← 3D training loop with SW validation
│   ├── evaluate_3d.py                ← per-subject 3D Dice evaluation on full val set
│   ├── visualize_3d.py               ← training curves, Dice distribution, overlay panels
│   │
│   ├── 3D/                           ← working copies from development branch
│   │   ├── dataset_3d.py
│   │   ├── model_3d.py
│   │   ├── train_3d.py
│   │   └── visualize_3d.py
│   │
│   ├── ── 2.5D Pipeline ────────────────────────────────────────────
│   ├── preprocess_msd_flair_t1gd_25d5.py          ← slice extraction and .npy saving
│   ├── build_runtime_metadata_flair_t1gd_25d5.py  ← metadata CSV for runtime NIfTI loading
│   ├── dataset.py                    ← BrainTumorSliceDataset: 5-slice 2D loading
│   ├── train.py                      ← 2.5D training loop (MONAI UNet2D)
│   ├── evaluate.py                   ← slice-level Dice evaluation with post-processing
│   ├── threshold_sweep.py            ← standalone threshold optimization
│   ├── plot_training_curves.py       ← 2.5D training curve figures
│   ├── inspect_predictions.py        ← visual inspection of 2.5D failure cases
│   ├── run_experiment.py             ← combined train + plot + eval runner
│   │
│   ├── ── Older preprocessing iterations ──────────────────────────
│   ├── preprocess_msd_flair_t1gd_25d.py  ← 2.5D 3-slice version
│   ├── preprocess_msd_flair_t1gd.py      ← FLAIR+T1gd no context
│   ├── preprocess_msd.py                 ← single-channel FLAIR (original)
│   └── inspect_msd.py                    ← dataset inspection utility
│
├── data/                             ← gitignored
│   ├── raw/
│   │   └── Task01_BrainTumour/
│   │       ├── imagesTr/             ← 484 multi-channel NIfTI images (.nii.gz)
│   │       └── labelsTr/             ← 484 segmentation label files (.nii.gz)
│   └── processed_3d/
│       ├── images/                   ← per-subject float32 volumes (2, 240, 240, 155)
│       ├── masks/                    ← per-subject uint8 binary masks (240, 240, 155)
│       └── metadata.csv              ← subject IDs, paths, tumor voxel counts, shapes
│
└── outputs/                          ← gitignored
    └── 3d_unet_v1/
        ├── checkpoints/
        │   ├── best_model_3d_unet_v1.pt   ← saved whenever SW val Dice improved
        │   └── latest_model_3d_unet_v1.pt ← saved every epoch end
        ├── logs/
        │   ├── training_log.csv       ← per-epoch metrics (loss, dice, sw_val_dice, lr, time)
        │   ├── train_subjects.txt     ← frozen list of 388 training subject IDs
        │   └── val_subjects.txt       ← frozen list of 96 validation subject IDs
        ├── evaluation/
        │   ├── subject_metrics_3d.csv ← per-subject 3D Dice at threshold 0.5
        │   └── threshold_sweep_3d.csv ← mean/median/std Dice across thresholds
        └── visualizations/
            ├── training_curves.png
            ├── dice_distribution.png
            └── overlays/
                ├── best/              ← top 10 subjects by Dice (green title)
                ├── worst/             ← bottom 10 subjects by Dice (red title)
                └── random/            ← 10 randomly sampled middle-tier subjects (gold title)
```

---

## 3. Dataset

**Medical Segmentation Decathlon — Task01_BrainTumour**  
Source: [http://medicaldecathlon.com](http://medicaldecathlon.com)

- **484 training subjects** with multi-channel MRI and voxel-level segmentation labels
- Each image: shape `(240, 240, 155, 4)` — four co-registered MRI modalities per voxel
- Each label: shape `(240, 240, 155)` — integer class map (0 = background, 1/2/3 = tumor subtypes)

| Channel Index | Modality | Clinical Role |
|:---:|---|---|
| 0 | FLAIR | Highlights tumor and edema; suppresses CSF signal |
| 1 | T1w | Anatomical reference; white/gray matter contrast |
| 2 | T1gd | Gadolinium contrast; shows active tumor and BBB breakdown |
| 3 | T2w | Sensitive to fluid and edema extent |

This project uses **channels 0 (FLAIR) and 2 (T1gd)** based on ablation experiments showing they provide the strongest segmentation signal as a pair.

Labels are **binarized**: all non-zero classes (necrosis, enhancing tumor, edema) are merged into a single whole-tumor class (`mask = label > 0`). This is the standard whole-tumor evaluation criterion used in BraTS benchmarks.

**Train/val split:** 388 training subjects / 96 validation subjects (80/20). The split is **frozen** in `outputs/.../logs/train_subjects.txt` and `val_subjects.txt` so all experiments remain directly comparable.

---

## 4. Environment Setup

### Requirements

```
Python 3.10+
PyTorch 2.x (CUDA 12.x recommended)
CUDA-capable GPU (tested on NVIDIA RTX 3090, 24 GB VRAM)
Windows 11
```

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai nibabel numpy pandas matplotlib tqdm scipy
pip install "napari[all]"
```

`napari` is only required for `4-channel_NIfTI_rendering.py`. All training, evaluation, and `visualize_3d.py` work without it.

### Download the dataset

1. Download Task01_BrainTumour from [http://medicaldecathlon.com](http://medicaldecathlon.com)
2. Extract to `data/raw/Task01_BrainTumour/`
3. Confirm structure: `data/raw/Task01_BrainTumour/imagesTr/` and `labelsTr/` each contain 484 `.nii.gz` files

---

## 5. Data Visualization

Two complementary tools are used to inspect raw MRI volumes and segmentation labels.

### ITK-SNAP — Slice-by-slice QA on raw NIfTI

**Download:** https://www.itksnap.org  
**License:** GPL (free)  
**Platform:** Windows 64-bit installer

ITK-SNAP is the fastest way to browse the raw data and verify that ground-truth labels visually align with the anatomy.

**Loading a subject with its segmentation:**

1. Open ITK-SNAP
2. `File → Open Main Image` → select `imagesTr/BRATS_001.nii.gz`
3. Use the time-point slider at the bottom to switch between the four modalities (T1, T1ce, T2, FLAIR) — the 4D volume's fourth axis is treated as a time series
4. `Segmentation → Load Segmentation` → select `labelsTr/BRATS_001.nii.gz`
5. The label overlay appears in all three orthogonal views (axial / coronal / sagittal)

**What to look for:**
- The tumor label (colored region) should sit within the bright area on FLAIR and on the enhancing region on T1gd
- Scrolling axially through depth (z) axis confirms label coverage across slices
- Misalignments or blank label regions indicate files to investigate further

---

### napari — Programmatic viewing of raw NIfTI and preprocessed .npy arrays

**Install:** `pip install "napari[all]"`  
**Docs:** https://napari.org  
**License:** BSD (free)

napari is used in this repo via `src/4-channel_NIfTI_rendering.py` to load all four MRI modalities simultaneously as independently toggleable layers, with the segmentation rendered as a discrete label overlay.

**Running the viewer:**

```bash
cd "e:\Brain Lesion Detection"
python src/4-channel_NIfTI_rendering.py
```

**What the script does:**

```python
import nibabel as nib
import napari

img = nib.load("data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
arr = img.get_fdata()          # shape: (240, 240, 155, 4) — full float64

modalities = ["T1", "T1ce", "T2", "FLAIR"]
viewer = napari.Viewer()

for i, name in enumerate(modalities):
    viewer.add_image(arr[..., i], name=name, colormap="gray", visible=(i == 3))

seg = nib.load("data/raw/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz")
viewer.add_labels(seg.get_fdata().astype(int), name="seg")

napari.run()
```

The viewer opens with only FLAIR visible by default. Toggle modalities by clicking the eye icon next to each layer name in the left panel. The `seg` layer renders each label class in a distinct color.

**Viewing preprocessed .npy volumes** (the FLAIR + T1gd stacks in `data/processed_3d/`):

```python
import numpy as np
import napari

arr = np.load("data/processed_3d/images/BRATS_001.npy")   # (2, H, W, D)
mask = np.load("data/processed_3d/masks/BRATS_001.npy")   # (H, W, D)

viewer = napari.Viewer()
viewer.add_image(arr[0], name="FLAIR (normalized)", colormap="gray")
viewer.add_image(arr[1], name="T1gd (normalized)",  colormap="magma", opacity=0.6)
viewer.add_labels(mask.astype(int), name="binary mask")
napari.run()
```

**Useful napari controls:**
- Scroll wheel: step through axial slices
- `Ctrl + scroll`: zoom
- Click layer name to select; use opacity slider to blend
- `V`: toggle visibility of selected layer
- `F`: fullscreen

---

## 6. Pipeline Overview

The full end-to-end data flow, from raw NIfTI files to poster figures:

```
Raw NIfTI (data/raw/)
        │
        ├──► 4-channel_NIfTI_rendering.py  (optional: visual QA via napari)
        │
        ▼  preprocess_3d.py
        │  Extract FLAIR (ch 0) + T1gd (ch 2) → normalize per channel
        │  → binarize labels → save (2,240,240,155) images and (240,240,155) masks
        ▼
data/processed_3d/  (.npy files + metadata.csv)
        │
        ▼  train_3d.py  ←  dataset_3d.py  ←  model_3d.py
        │  Sample random 128³ patches (50% foreground-biased)
        │  Feed batches of patches through 3D U-Net
        │  Compute BCE + Dice loss → backprop → Adam + cosine LR
        │  Per-epoch: sliding-window 3D Dice on 10 val subjects
        ▼
outputs/3d_unet_v1/checkpoints/  (best_model.pt and latest_model.pt)
        │
        ▼  evaluate_3d.py  ←  sliding_window.py
        │  Full sliding-window inference on all 96 val subjects
        │  Per-subject 3D Dice, threshold sweep [0.3 – 0.7]
        ▼
outputs/3d_unet_v1/evaluation/  (CSVs)
        │
        ▼  visualize_3d.py
        │  Training curves, Dice distribution, best/worst/random overlay panels
        ▼
outputs/3d_unet_v1/visualizations/
```

---

## 7. 2.5D Pipeline

The 2.5D pipeline treats the 3D brain volume as a stack of 2D axial slices. Each training sample is a stack of 5 adjacent FLAIR slices + 5 adjacent T1gd slices centered on the target slice, forming a 10-channel 2D input to a standard 2D U-Net.

**Run order:**

```bash
cd src
# Option A: pre-save slices to .npy
python preprocess_msd_flair_t1gd_25d5.py

# Option B: build metadata only, load NIfTI at runtime
python build_runtime_metadata_flair_t1gd_25d5.py

python train.py
python evaluate.py
python plot_training_curves.py
```

**Key files:**

| File | Role |
|------|------|
| `preprocess_msd_flair_t1gd_25d5.py` | Extracts and saves 2D slices with 5-slice context as `.npy` |
| `build_runtime_metadata_flair_t1gd_25d5.py` | Builds metadata CSV pointing at original NIfTI (no pre-saving) |
| `dataset.py` | `BrainTumorSliceDataset` — loads NIfTI at runtime, returns 10-channel stacks |
| `train.py` | MONAI 2D UNet, BCE + Tversky loss, small-lesion oversampling |
| `evaluate.py` | Per-slice Dice with 3D connected-component post-processing and threshold sweep |
| `inspect_predictions.py` | Visual panel tool for examining failure cases subject-by-subject |

**Note on metrics:** The 2.5D pipeline reports **per-slice Dice**, then aggregates to subject level by reconstructing the volume from slice predictions. This is fundamentally different from the 3D pipeline's whole-volume Dice and the numbers are **not directly comparable** — they measure different things.

---

## 8. 3D Pipeline

### Step 1: Preprocessing

```bash
cd src
python preprocess_3d.py
```

**What it does:**

1. Reads every `.nii.gz` in `data/raw/Task01_BrainTumour/imagesTr/`
2. Skips macOS artifact files (prefixed `._`) and warns on missing labels
3. Extracts FLAIR (channel index 0) and T1gd (channel index 2) from the 4D volume
4. For each channel independently:
   - Casts to `float32`, replaces NaN/Inf with 0
   - Identifies foreground voxels where intensity > 0
   - Clips to the 1st–99th percentile of foreground voxels (removes rare outliers)
   - Min-max rescales to approximately [0, 1]
5. Binarizes the label: `mask = (label > 0).astype(uint8)`
6. Stacks channels: `np.stack([flair, t1gd], axis=0)` → shape `(2, H, W, D)` = `(2, 240, 240, 155)`
7. Saves per subject:
   - `data/processed_3d/images/{subject_id}.npy` — float32, `(2, 240, 240, 155)`
   - `data/processed_3d/masks/{subject_id}.npy` — uint8, `(240, 240, 155)`
8. Writes `data/processed_3d/metadata.csv` with columns: `subject_id`, `image_path`, `mask_path`, `tumor_voxels`, `shape_h`, `shape_w`, `shape_d`

**Expected output:** 484 image files, 484 mask files, 1 CSV. Approximate runtime: 13–15 minutes.

**Axis convention:** Arrays are saved as `(C, H, W, D)` — matching the NIfTI voxel order from `get_fdata()` with a channel axis prepended. The depth axis D=155 is the last spatial dimension. All downstream code in `dataset_3d.py` and `sliding_window.py` expects and indexes this layout consistently.

**Note on normalization:** The 3D branch uses **percentile clip + min-max** normalization (output ≈ [0, 1]), which differs from the 2.5D pipeline's **z-score** normalization (mean 0, std 1). This is intentional and should be noted in any methods section comparing the two pipelines.

---

### Step 2: Training

```bash
cd src
python train_3d.py
```

**Key configuration constants** (edit at the top of `train_3d.py`):

| Constant | Default | Meaning |
|----------|---------|---------|
| `EXPERIMENT_NAME` | `"3d_unet_v1"` | Output directory name under `outputs/` |
| `PATCH_SIZE` | `(128, 128, 128)` | Spatial size of each training patch in voxels |
| `PATCHES_PER_SUBJECT` | `4` | Random patches sampled per subject per epoch |
| `BATCH_SIZE` | `2` | Patches processed per GPU forward pass |
| `NUM_WORKERS` | `2` | CPU workers for DataLoader (increase to 6–8 on high-RAM machines) |
| `EPOCHS` | `50` | Total training epochs |
| `LR` | `1e-4` | Starting learning rate (cosine-annealed to 1e-6) |
| `GRAD_CLIP_NORM` | `1.0` | Max gradient norm before clipping |
| `VAL_SW_SUBJECTS` | `10` | Subjects to run full sliding-window eval on per epoch |
| `SW_OVERLAP` | `0.5` | Fraction of patch overlap during sliding-window inference |
| `LOSS_BCE_WEIGHT` | `0.5` | Weight of Binary Cross-Entropy in combined loss |
| `LOSS_DICE_WEIGHT` | `0.5` | Weight of Dice loss in combined loss |

**What each epoch does:**

1. **Train phase:** for each of the 388 training subjects, sample `PATCHES_PER_SUBJECT` random 128³ patches (50% centered on a tumor voxel, 50% at a random location). Pack into batches of 2, run forward pass, compute combined loss, backpropagate, clip gradients to max norm 1.0, update weights.
2. **Validation patch phase:** same patch extraction on the 96 val subjects, no gradient updates. Computes validation loss and patch-level Dice as a fast training signal.
3. **Sliding-window validation phase:** for `VAL_SW_SUBJECTS` complete validation subjects, tiles 128³ patches across the full 240×240×155 volume with 50% overlap, averages probabilities in overlapping regions, thresholds at 0.5, computes full-volume 3D Dice against ground truth. This is the metric that determines whether `best_model.pt` is saved.
4. **Checkpoint and log:** write `training_log.csv`, save `latest_model.pt` unconditionally, save `best_model.pt` only if sliding-window Dice improved.

**Subject split:** the script first looks for frozen split files from the 2.5D baseline run at `outputs/runtime_flair_t1gd_25d5_smalllesion_baseline_v1/logs/`. If those files exist, they are used to guarantee an identical 388/96 split across both pipelines. If absent, the script falls back to a deterministic 80/20 random split using `RANDOM_SEED=42`.

**Resuming after a crash:** the script does not support automatic resume. To restart from a checkpoint, add a `torch.load` call before the training loop (loading `latest_model.pt`) and change `range(EPOCHS)` to `range(last_completed_epoch, EPOCHS)`. The `training_log.csv` records which epochs completed successfully — the last row's `epoch` value is the authoritative restart point.

**Performance recommendations for high-RAM systems (128 GB DDR5 / i9-14700KF):**

```python
NUM_WORKERS = 8          # was 2
# In DataLoader calls, add:
prefetch_factor = 4      # was default 2
VAL_SW_SUBJECTS = 5      # was 10; reduces per-epoch wall time, no effect on model
```

---

### Step 3: Evaluation

```bash
cd src
python evaluate_3d.py
```

Runs **full** sliding-window inference on all 96 validation subjects using `best_model_3d_unet_v1.pt`. For each subject:

1. Loads the full `(2, 240, 240, 155)` volume from `data/processed_3d/images/`
2. Tiles 128³ patches across the volume with 50% overlap
3. Averages sigmoid probabilities in overlapping regions to produce a smooth probability map
4. Thresholds at each value in `[0.3, 0.4, 0.5, 0.6, 0.7]`
5. Computes full-volume 3D Dice against the binary ground-truth mask

**Outputs:**
- `evaluation/subject_metrics_3d.csv` — per-subject 3D Dice at threshold 0.5, plus tumor voxel count
- `evaluation/threshold_sweep_3d.csv` — mean, median, and std Dice at each threshold

**Checkpoint resolution:** uses `best_model_3d_unet_v1.pt` if it exists; falls back to `latest_model_3d_unet_v1.pt`. Uses the same frozen val split as training.

---

### Step 4: Visualization

```bash
cd src
python visualize_3d.py
```

Produces three types of figures saved under `outputs/3d_unet_v1/visualizations/`:

**1. Training curves** (`training_curves.png`)  
Two side-by-side panels: loss convergence (train + val) and Dice progression (train patch, val patch, SW 3D). The best-epoch annotation marks when `best_model.pt` was saved.

**2. Dice distribution** (`dice_distribution.png`)  
Horizontal bar chart of all 96 val subjects sorted by 3D Dice ascending. Bars below 0.80 are red, above are cyan. Mean and median lines annotated.

**3. Overlay panels** (`overlays/{best,worst,random}/`)  
For each of 30 selected subjects (10 per category), a 5-panel figure showing:

| Panel | Content |
|-------|---------|
| FLAIR | Raw normalized grayscale at the axial slice with maximum lesion area |
| T1gd | Same axial slice from the contrast channel |
| GT mask | FLAIR slice with ground-truth tumor in **cyan** |
| Prediction | FLAIR slice with model prediction in **pink** |
| Probability | FLAIR slice with sigmoid probability as a hot colormap (dark = low, white = high) |

- **best/** — top 10 subjects by Dice (green title)
- **worst/** — bottom 10 subjects by Dice (red title)
- **random/** — 10 randomly sampled from the middle 76 subjects (gold title)

---

## 9. Results

### 3D U-Net — Final Evaluation (96 validation subjects)

| Metric | Value |
|--------|-------|
| Mean 3D Dice @ threshold 0.5 | **0.894** |
| Median 3D Dice @ threshold 0.5 | **0.923** |
| Std 3D Dice | 0.076 |
| Subjects with Dice > 0.87 | 77 / 96 (80%) |
| Best subject (BRATS_442) | 0.968 |
| Worst subject (BRATS_381) | 0.555 — small tumor (~27k voxels) |

### Threshold Sweep

| Threshold | Mean Dice | Median Dice | Std |
|-----------|-----------|-------------|-----|
| 0.30 | 0.891 | 0.915 | 0.077 |
| 0.40 | 0.894 | 0.922 | 0.076 |
| **0.50** | **0.894** | **0.923** | **0.076** |
| 0.60 | 0.893 | 0.922 | 0.077 |
| 0.70 | 0.891 | 0.919 | 0.079 |

Threshold 0.5 is the optimal operating point. The near-flat curve across 0.4–0.6 indicates a well-calibrated model with no strong systematic over- or under-prediction bias.

### Training Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 45 of 50 (hardware crash at epoch 43; fully converged) |
| LR at epoch 45 | ~3.4e-6 (cosine floor; final 5 epochs would not have changed results) |
| Best SW Val Dice (10 subjects/epoch) | **0.9198** at epoch 45 |
| Approximate epoch wall time | ~160 seconds |
| Total training time | ~2 hours |

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CPU | Intel Core i9-14700KF |
| RAM | 128 GB DDR5 |
| OS | Windows 11 |

---

## 10. Glossary of All Terms

### MRI Modalities

**FLAIR (Fluid-Attenuated Inversion Recovery)**  
An MRI sequence that nulls the signal from cerebrospinal fluid (the water that surrounds and cushions the brain). In a standard T2 scan, CSF appears bright, which can obscure lesions near the brain surface. FLAIR suppresses that signal, making tumors and edema appear bright against a dark background. It is the most diagnostically informative channel for brain tumor detection and is the primary channel in this pipeline.

**T1gd (T1-weighted with Gadolinium Contrast)**  
Gadolinium is a paramagnetic contrast agent injected intravenously before scanning. It does not normally cross the blood-brain barrier (BBB). Where a tumor has broken down the BBB — as aggressive tumors do — gadolinium accumulates and causes that tissue to appear bright white on the T1-weighted scan. This reveals active, enhancing tumor tissue and helps distinguish it from surrounding edema or dead tissue.

**T1w (T1-weighted)**  
Anatomical reference scan. Fat and white matter appear bright, gray matter is medium, CSF is dark. Used for structural reference; less diagnostically useful for tumor detection than FLAIR or T1gd.

**T2w (T2-weighted)**  
Water appears bright, so edema, inflammation, and CSF are all bright. Useful for detecting the spatial extent of tumor-related swelling but less specific than FLAIR.

---

### Data

**NIfTI (.nii.gz)**  
Neuroimaging Informatics Technology Initiative format. A gzip-compressed binary file storing a multi-dimensional array of voxel intensities alongside a header describing physical dimensions (voxel size in mm), coordinate system orientation, and acquisition parameters. The standard format for research-grade brain MRI data.

**Voxel**  
The 3D equivalent of a pixel. One voxel represents a small cube of tissue. In MSD Task01, each voxel is approximately 1 mm × 1 mm × 1 mm. The full brain volume is 240 × 240 × 155 voxels, covering roughly the entire head.

**Ground truth mask (GT)**  
The manually annotated binary map of tumor location, produced by expert neuroradiologists drawing contours around visible tumor on each axial slice. Used as both the training target and evaluation reference. "Binary" here means all non-zero tumor subtypes (necrosis = 1, enhancing tumor = 2, edema = 3) are merged: `mask = (label > 0)`.

**Binarization**  
Converting multi-class labels {0, 1, 2, 3} into a single binary class {0, 1}. Applied as `mask = (label > 0)` so the model predicts "is there any tumor here?" rather than "which subtype of tumor is here?". This is the standard whole-tumor criterion used in BraTS benchmarks and clinical whole-lesion analysis.

**Axial slice**  
A 2D cross-section through the brain volume at a fixed depth (z) index. Axial view = looking down from above, like cross-sections through a loaf of bread. The dataset has 155 such slices per subject.

---

### Preprocessing

**Normalization (p1/p99 min-max)**  
MRI intensity values are uncalibrated — the same tissue type can produce different raw numbers across scanners, field strengths, or acquisition settings. To make all volumes comparable, for each channel: (1) identify foreground voxels with intensity > 0, (2) clip the 1st and 99th percentile to remove rare extreme outliers, (3) rescale to [0, 1]. Applied independently per channel per subject.

**Z-score normalization** (2.5D branch only)  
Alternative used in the runtime 2.5D dataset: after percentile clipping, subtract the foreground mean and divide by the foreground standard deviation. Output is approximately mean 0, std 1 rather than [0, 1]. This is a standard MRI normalization in supervised learning but differs from the 3D branch; do not directly compare raw intensity values between the two preprocessed datasets.

**Foreground voxels**  
Voxels with intensity > 0. In brain MRI, voxels outside the head (air, scanner background) are zero. Normalization statistics are computed only on foreground to avoid background noise distorting the percentile estimates.

**Axis convention (C, H, W, D)**  
How array dimensions are ordered. In this codebase: C = channels (2: FLAIR and T1gd), H = height (240), W = width (240), D = depth (155). Arrays are saved and loaded in this order. PyTorch expects input as `(Batch, C, D1, D2, D3)`; at training time a batch of two patches becomes shape `(2, 2, 128, 128, 128)`.

---

### Model Architecture

**U-Net**  
A convolutional neural network architecture originally designed for biomedical image segmentation (Ronneberger et al., 2015). Consists of an encoder (contracting path) that downsamples the input to capture global context, a bottleneck at maximum compression, and a decoder (expansive path) that upsamples back to original resolution. Skip connections transfer spatial detail from each encoder stage to the corresponding decoder stage. The name comes from the U-shape of the architecture diagram.

**3D U-Net**  
The volumetric extension of U-Net where every `Conv2d`, `MaxPool2d`, and `ConvTranspose2d` is replaced with its 3D counterpart (`Conv3d`, `MaxPool3d`, `ConvTranspose3d`). Processes the full spatial context of a 3D volume simultaneously, capturing anatomical relationships across all three axes that would be lost when treating slices independently.

**Encoder**  
The downsampling half. Each of 4 stages applies two 3×3×3 convolutions, then a 2×2×2 max pool that halves each spatial dimension. Feature channels grow: 2 → 32 → 64 → 128 → 256. Captures increasingly abstract, global context.

**Decoder**  
The upsampling half. Each stage uses a transposed convolution to double spatial dimensions, concatenates the skip connection from the matching encoder stage, then applies two convolutions. Combines global context from deep layers with fine spatial detail from shallow layers to reconstruct precise voxel-level predictions.

**Skip connection**  
A direct path from an encoder layer to the corresponding decoder layer at the same spatial resolution. Preserves fine-grained feature maps (edges, textures, local anatomy) that would otherwise be lost through the bottleneck. Essential for precise segmentation boundaries.

**Bottleneck**  
The deepest layer between encoder and decoder. Volume is downsampled by 8× (for 3 pooling layers) or more; feature channels are at their maximum (512 in this model). Captures the most global, abstract representation of the input.

**InstanceNorm3d**  
Normalizes activations per-sample, per-channel — independently for every (sample, channel) pair in the batch. Preferred over BatchNorm3d in 3D medical segmentation because batch sizes are very small (1–2 patches) and BatchNorm's learned statistics become unreliable at small batches. InstanceNorm works correctly at batch size 1.

**LeakyReLU(0.01)**  
Activation function that passes positive values unchanged and multiplies negative values by 0.01 rather than clamping to 0. The small negative slope ("leak") prevents dead neurons during backpropagation and is standard in 3D segmentation networks.

**Feature sizes [32, 64, 128, 256]**  
The channel count at each encoder depth level. Intentionally smaller than full-capacity 3D U-Nets (which might use [64, 128, 256, 512]) to fit within 24 GB VRAM at batch size 2 with 128³ patches and AMP enabled.

---

### Training

**Patch**  
A 128×128×128 sub-volume cropped from the full 240×240×155 brain volume. Required because the full volume cannot fit in GPU VRAM during a forward pass with a full 3D U-Net (a single float32 volume with 2 channels = ~55 MB; model activations during the forward pass multiply this by 10–20×). During training the model only ever sees patches; during evaluation, sliding-window inference assembles predictions from overlapping patches to cover the full volume.

**Foreground-biased patch sampling (50/50)**  
When selecting a patch center: 50% of samples are centered on a randomly chosen tumor voxel; 50% are at a completely random location. Without this bias, brain tumors occupy under 5% of brain volume on average, so most random patches contain no tumor. A model trained on such patches learns that predicting all-background incurs low loss — a failure mode called class imbalance collapse. The 50/50 split guarantees the model sees enough positive examples to learn to identify tumor.

**Epoch**  
One complete pass through the training dataset. With 388 training subjects × 4 patches per subject = 1,552 training patches per epoch, packed into batches of 2 = 776 gradient update steps per epoch.

**Batch size**  
Number of patches processed simultaneously in a single forward + backward pass on the GPU. Larger batches use more VRAM but typically produce more stable gradient estimates and improve GPU utilization. At batch size 2 with 128³ patches and AMP enabled on the RTX 3090, model weights occupy ~0.4 GB VRAM (activations during the forward pass consume considerably more).

**Loss function**  
The scalar value the optimizer minimizes. Two complementary terms are combined at 50/50 weight:

- **Binary Cross-Entropy (BCE):** for each voxel independently, computes `−[y · log(p) + (1−y) · log(1−p)]` where `y ∈ {0,1}` is the ground truth and `p ∈ (0,1)` is the predicted probability. Penalizes confident wrong predictions most heavily. Ensures per-voxel accuracy.
- **Dice loss:** `1 − 2·|Pred ∩ GT| / (|Pred| + |GT|)`. Measures overlap of the predicted region against ground truth as a whole. Penalizes size mismatches — if the model predicts a blob half the correct size, Dice loss is high even if all predicted voxels are correct. Essential for class-imbalanced data.

Combined: `total_loss = 0.5 × BCE + 0.5 × Dice loss`

**Backpropagation**  
The algorithm that computes the gradient of the loss with respect to every weight in the network by applying the chain rule backwards from the output layer. These gradients tell the optimizer how to adjust each weight to reduce the loss.

**Adam optimizer**  
Adaptive moment estimation. Maintains per-parameter momentum (exponential moving average of past gradients) and per-parameter adaptive learning rate scaling (based on past gradient magnitudes). Converges faster and more robustly than vanilla SGD. The standard choice for medical image segmentation.

**Learning rate (LR)**  
Scales how large each weight update is. Too high: training diverges. Too low: training is unnecessarily slow. Starting at 1e-4 is a conservative, well-established default for 3D segmentation networks.

**Cosine annealing scheduler**  
Reduces the learning rate following a cosine curve from the initial value (1e-4) to a minimum (1e-6) over the total epoch count. High LR in early epochs enables fast coarse learning; low LR in later epochs enables fine-grained tuning without overshooting. At epoch 45 the LR had annealed to ~3.4e-6 — effectively at the floor — which is why the remaining 5 epochs after the hardware crash would not have changed performance.

**Gradient clipping (`clip_grad_norm_ = 1.0`)**  
Before each weight update, computes the Euclidean norm of all gradients across the model. If this norm exceeds 1.0, all gradients are scaled down proportionally. Prevents "exploding gradients" — a failure mode particularly common early in 3D network training where a single high-loss batch can cause catastrophically large updates that destroy accumulated learning.

**AMP (Automatic Mixed Precision)**  
Uses 16-bit float (float16) for most computations instead of 32-bit (float32), roughly halving VRAM usage and accelerating GPU arithmetic. PyTorch handles conversion automatically and maintains a float32 master copy of weights to prevent precision loss during updates.

**GradScaler**  
The AMP component that prevents numerical underflow during float16 backpropagation. Multiplies the loss by a large scale factor before backward pass (so gradients are representable in float16), then scales them back down before the optimizer step. Automatically reduces the scale factor if NaN or Inf gradients are detected.

**`persistent_workers`**  
DataLoader option that keeps CPU worker processes alive between epochs instead of re-spawning them. Reduces inter-epoch overhead on long runs at the cost of holding loaded workers in RAM between epochs.

**Augmentation**  
Random geometric transformations applied to training patches to artificially increase variety and reduce overfitting. Implemented in `dataset_3d.py`: random flips along each of the three spatial axes, each applied with 50% probability independently. Label-preserving — the mask receives identical flips as the image.

---

### Evaluation

**Dice coefficient**  
The primary evaluation metric. Measures volumetric overlap between two binary regions:

```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```

Range: 0 (no overlap at all) to 1 (perfect agreement). A Dice of 0.923 means approximately 92.3% volumetric agreement between the model's predicted tumor region and the radiologist-annotated ground truth across the entire 3D volume.

**3D Dice vs per-slice Dice**  
Not the same metric and not directly comparable. Per-slice Dice computes 2D overlap on each axial slice independently, then averages across all 155 slices. Slices with no tumor in either prediction or GT contribute Dice = 1.0 (two empty sets perfectly agree), inflating the average. 3D Dice computes overlap across the entire volume in one calculation, treating every voxel equally, with no inflation from empty slices. 3D Dice is stricter and more clinically meaningful. A per-slice Dice of 0.85 and a whole-volume 3D Dice of 0.92 are not in conflict — they simply measure different things.

**Threshold**  
The model's final layer outputs a raw logit (unbounded real number). Sigmoid converts this to a probability p ∈ [0, 1]. The threshold converts that probability map to a binary mask: voxels with `p ≥ threshold` are classified as tumor. At threshold 0.5: the model must be more than 50% confident to predict tumor. Lower thresholds increase sensitivity (more tumor detected, more false positives); higher thresholds increase specificity (fewer false positives, risk of missing small lesions). The threshold sweep confirmed 0.5 is optimal for this model.

**Sliding-window inference**  
Technique for applying a patch-trained model to a larger volume. A 128³ window slides across the full 240×240×155 volume at fixed stride intervals. At each position, the model predicts a probability map. Because adjacent windows overlap by 50%, most voxels are predicted multiple times from slightly different receptive field contexts. The final probability at each voxel is the average of all predictions that included it. This averaging smooths boundary artifacts at patch edges and produces better predictions than non-overlapping tiling.

**SW Val Dice (`sw_val_dice`)**  
Sliding-window 3D Dice computed on a subset of full validation volumes during training. This is the metric used to select which epoch's weights become `best_model.pt`. Monitored every epoch on `VAL_SW_SUBJECTS` (default 10) subjects for speed; the definitive full-set evaluation runs separately via `evaluate_3d.py`.

**Patch Dice (`train_dice` / `val_dice`)**  
Dice computed on individual 128³ patches rather than full volumes. Because empty patches (both prediction and GT are all-zero) contribute Dice ≈ 1.0, and because 50/50 foreground-biased sampling makes patches artificially tumor-heavy relative to real anatomy, this metric is biased upward compared to whole-volume Dice. Use it as a fast signal that training is progressing, not as a clinical performance number.

---

## 11. Understanding the Training Graphs

Produced by `visualize_3d.py` and saved to `outputs/3d_unet_v1/visualizations/training_curves.png`.

### Left Panel — Loss Curve

X-axis: epoch number (1–45). Y-axis: combined BCE + Dice loss (lower = better).

- **Red (Train loss):** computed on training patches with gradient updates. Expected to decrease throughout training.
- **Cyan (Val loss):** computed on held-out validation patches with no updates. Indicates whether the model generalizes or memorizes.

Healthy training: both curves decrease together, with val loss slightly above train loss. If val loss increases while train loss continues to decrease, the model is overfitting — memorizing training data rather than learning generalizable features.

Your training shows clean convergence: both curves drop sharply in epochs 1–15, then plateau together from epoch 25 onward. No overfitting signal. The parallel plateau means the model stopped improving, not that it started memorizing.

### Right Panel — Dice Curve

- **Red (Train Dice, patch):** patch-level hard Dice at threshold 0.5 on training patches. Inflated by empty-patch Dice = 1.0 contributions. Shows rapid improvement in early epochs.
- **Cyan (Val Dice, patch):** same on validation patches. Same inflation caveat.
- **Green dashed (SW Val Dice, 3D):** full sliding-window volumetric Dice on 10 validation subjects per epoch. The real performance metric. Starts at ~0.75 at epoch 1 (the model rapidly learns basic anatomy), rises to ~0.92 by epoch 25, then plateaus. The **gold dotted vertical line** marks the best epoch (epoch 45, Dice 0.9198) where `best_model.pt` was saved.

The gap between patch Dice (~0.83) and SW Val Dice (~0.92) is expected and does not indicate a discrepancy. Patch Dice is pulled downward by the 50/50 foreground sampling (patches are artificially half tumor whereas real brains are ~5% tumor). SW Dice is the clean whole-volume metric.

---

## 12. File Reference

All files are on `main` (merged from `3d-unet` branch via PRs #1, #2, #3).

| File | Pipeline | Purpose | Typical run order |
|------|----------|---------|:-----------------:|
| `4-channel_NIfTI_rendering.py` | both | napari viewer for raw 4-channel NIfTI + label overlay | — |
| `preprocess_msd.py` | 2.5D | Single-channel FLAIR slice extraction (original baseline) | — |
| `preprocess_msd_flair_t1gd.py` | 2.5D | FLAIR + T1gd slice extraction (v1) | — |
| `preprocess_msd_flair_t1gd_25d.py` | 2.5D | 2.5D 3-slice context preprocessing | — |
| `preprocess_msd_flair_t1gd_25d5.py` | 2.5D | 2.5D 5-slice context (current 2.5D) | 1 |
| `build_runtime_metadata_flair_t1gd_25d5.py` | 2.5D | Metadata CSV for runtime NIfTI loading | 1 (alt) |
| `dataset.py` | 2.5D | `BrainTumorSliceDataset`: 10-channel 2D patch loading | — |
| `train.py` | 2.5D | Training loop: MONAI UNet2D, BCE + Tversky, small-lesion oversampling | 2 |
| `evaluate.py` | 2.5D | Per-slice Dice with post-processing and threshold sweep | 3 |
| `threshold_sweep.py` | 2.5D | Standalone threshold optimization over saved predictions | — |
| `plot_training_curves.py` | 2.5D | 2.5D training curve figures | 4 |
| `inspect_predictions.py` | 2.5D | Visual subject-level inspection of 2.5D failure cases | — |
| `inspect_msd.py` | both | Dataset inspection utility (shapes, label distributions) | — |
| `run_experiment.py` | 2.5D | Combined train + plot + eval runner | — |
| `preprocess_3d.py` | 3D | NIfTI → normalized .npy volumes + metadata CSV | 1 |
| `dataset_3d.py` | 3D | `BrainTumorDataset3D`: foreground-biased 128³ patch sampling | — |
| `model_3d.py` | 3D | `UNet3D`: 3D encoder-decoder, InstanceNorm3d, [32,64,128,256] features | — |
| `sliding_window.py` | 3D | Overlapping patch inference → full-volume probability map | — |
| `train_3d.py` | 3D | 3D training loop: BCE + Dice, AMP, cosine LR, grad clip, SW val | 2 |
| `evaluate_3d.py` | 3D | Full val-set 3D Dice evaluation + threshold sweep CSVs | 3 |
| `visualize_3d.py` | 3D | Training curves, Dice distribution, best/worst/random overlay panels | 4 |
| `3D/` | 3D | Working copies from development branch (reference only) | — |
