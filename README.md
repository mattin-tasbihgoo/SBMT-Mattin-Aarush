# Brain Lesion Detection — AI-Based Neuroimaging Segmentation Pipeline

> **Context:** This repository is the neuroimaging component of a broader research project on AI-based prediction of chemotherapy-induced cognitive impairment (CICI), presented at the **SBMT 2026 NeuroTech Annual Convention**, Los Angeles, April 16–19.

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Environment Setup](#4-environment-setup)
5. [Pipeline Overview](#5-pipeline-overview)
6. [2.5D Pipeline (main branch)](#6-25d-pipeline-main-branch)
7. [3D Pipeline (3d-unet branch)](#7-3d-pipeline-3d-unet-branch)
   - [Step 1: Preprocessing](#step-1-preprocessing)
   - [Step 2: Training](#step-2-training)
   - [Step 3: Evaluation](#step-3-evaluation)
   - [Step 4: Visualization](#step-4-visualization)
8. [Results](#8-results)
9. [Glossary of All Terms](#9-glossary-of-all-terms)
10. [Understanding the Training Graphs](#10-understanding-the-training-graphs)
11. [File Reference](#11-file-reference)

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
| **3D U-Net (this branch)** | **3D volumetric** | **Mean 3D Dice 0.894 / Median 0.923** |

---

## 2. Repository Structure

```
Brain-Lesion-Detection/
├── src/
│   ├── preprocess_3d.py              ← 3D: NIfTI → normalized .npy volumes
│   ├── dataset_3d.py                 ← 3D: patch-based PyTorch Dataset
│   ├── model_3d.py                   ← 3D: U-Net architecture (Conv3d)
│   ├── sliding_window.py             ← 3D: full-volume inference via overlapping patches
│   ├── train_3d.py                   ← 3D: training loop
│   ├── evaluate_3d.py                ← 3D: per-subject 3D Dice evaluation
│   ├── visualize_3d.py               ← 3D: training curves, Dice distribution, overlays
│   │
│   ├── dataset.py                    ← 2.5D: runtime slice Dataset
│   ├── train.py                      ← 2.5D: training loop
│   ├── evaluate.py                   ← 2.5D: slice-level evaluation
│   ├── build_runtime_metadata_flair_t1gd_25d5.py  ← 2.5D: metadata builder
│   ├── preprocess_msd_flair_t1gd_25d5.py          ← 2.5D: slice preprocessing
│   ├── plot_training_curves.py       ← shared: training curve plots
│   ├── inspect_predictions.py        ← 2.5D: visual inspection tool
│   ├── threshold_sweep.py            ← 2.5D: threshold optimization
│   └── run_experiment.py             ← 2.5D: combined train+eval runner
│
├── data/                             ← gitignored
│   ├── raw/
│   │   └── Task01_BrainTumour/
│   │       ├── imagesTr/             ← 484 multi-channel NIfTI images
│   │       └── labelsTr/             ← 484 segmentation label files
│   └── processed_3d/
│       ├── images/                   ← per-subject .npy volumes (2, 240, 240, 155)
│       ├── masks/                    ← per-subject .npy masks (240, 240, 155)
│       └── metadata.csv
│
└── outputs/                          ← gitignored
    └── 3d_unet_v1/
        ├── checkpoints/
        │   ├── best_model_3d_unet_v1.pt
        │   └── latest_model_3d_unet_v1.pt
        ├── logs/
        │   ├── training_log.csv
        │   ├── train_subjects.txt
        │   └── val_subjects.txt
        ├── evaluation/
        │   ├── subject_metrics_3d.csv
        │   └── threshold_sweep_3d.csv
        └── visualizations/
            ├── training_curves.png
            ├── dice_distribution.png
            └── overlays/
                ├── best/
                ├── worst/
                └── random/
```

---

## 3. Dataset

**Medical Segmentation Decathlon — Task01_BrainTumour**  
Source: [http://medicaldecathlon.com](http://medicaldecathlon.com)

- **484 training subjects** with multi-channel MRI and voxel-level segmentation labels
- Each image: shape `(240, 240, 155, 4)` — four co-registered MRI modalities per voxel
- Each label: shape `(240, 240, 155)` — integer class map (0 = background, 1/2/3 = tumor subtypes)

| Channel Index | Modality | Clinical Role |
|---|---|---|
| 0 | FLAIR | Highlights tumor/edema; suppresses CSF |
| 1 | T1w | Anatomical reference |
| 2 | T1gd | Gadolinium contrast; shows active tumor and BBB breakdown |
| 3 | T2w | Sensitive to fluid and edema |

This project uses **channels 0 (FLAIR) and 2 (T1gd)** based on ablation experiments showing they provide the strongest segmentation signal as a pair.

Labels are **binarized**: all non-zero classes are merged into a single "whole tumor" class (`mask = label > 0`). This is the standard whole-tumor evaluation used in BraTS benchmarks.

**Train/val split:** 388 training subjects / 96 validation subjects (80/20). The split is **frozen** — stored in `outputs/.../logs/train_subjects.txt` and `val_subjects.txt` — so all experiments are directly comparable.

---

## 4. Environment Setup

### Requirements

```
Python 3.10+
PyTorch 2.x (CUDA 12.x recommended)
CUDA-capable GPU (tested on RTX 3090, 24 GB VRAM)
```

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai nibabel numpy pandas matplotlib tqdm scipy
```

### Download the dataset

1. Download Task01_BrainTumour from [http://medicaldecathlon.com](http://medicaldecathlon.com)
2. Extract to `data/raw/Task01_BrainTumour/`
3. Confirm structure: `data/raw/Task01_BrainTumour/imagesTr/` and `labelsTr/` each contain 484 `.nii.gz` files

---

## 5. Pipeline Overview

The full end-to-end data flow, from raw NIfTI files to poster figures:

```
Raw NIfTI (data/raw/)
        │
        ▼  preprocess_3d.py
        │  Extract FLAIR + T1gd → normalize per-channel → binarize labels
        │  Save (2, 240, 240, 155) images and (240, 240, 155) masks
        ▼
data/processed_3d/  (.npy files + metadata.csv)
        │
        ▼  train_3d.py  ←  dataset_3d.py  ←  model_3d.py
        │  Sample random 128³ patches (50% foreground-biased)
        │  Feed batches of patches through 3D U-Net
        │  Compute BCE + Dice loss, backprop, Adam + cosine LR
        │  Evaluate with sliding-window 3D Dice every epoch
        ▼
outputs/3d_unet_v1/checkpoints/  (best and latest .pt files)
        │
        ▼  evaluate_3d.py  ←  sliding_window.py
        │  Full sliding-window inference on all 96 val subjects
        │  Per-subject 3D Dice, threshold sweep CSV
        ▼
outputs/3d_unet_v1/evaluation/
        │
        ▼  visualize_3d.py
        │  Training curves, Dice distribution, overlay panels
        ▼
outputs/3d_unet_v1/visualizations/
```

---

## 6. 2.5D Pipeline (main branch)

The 2.5D pipeline treats the 3D brain volume as a stack of 2D axial slices. Each training sample is a stack of 5 adjacent FLAIR slices + 5 adjacent T1gd slices centered on the target slice, forming a 10-channel 2D input to a standard 2D U-Net.

**Key files:**
- `preprocess_msd_flair_t1gd_25d5.py` — extracts 2D slices and saves them as `.npy` files
- `build_runtime_metadata_flair_t1gd_25d5.py` — builds a metadata CSV pointing at original NIfTI files (runtime loading, no pre-saved slices)
- `dataset.py` — `BrainTumorSliceDataset`: loads NIfTI at runtime, returns 5-slice stacks
- `train.py` — MONAI 2D UNet, BCE + Tversky loss, small-lesion oversampling
- `evaluate.py` — per-slice Dice with 3D connected-component post-processing

**Note on metrics:** The 2.5D pipeline reports **per-slice Dice**, then aggregates to subject level by reconstructing the volume from slice predictions. This is fundamentally different from the 3D pipeline's whole-volume Dice and the numbers are **not directly comparable**.

---

## 7. 3D Pipeline (3d-unet branch)

### Step 1: Preprocessing

```bash
cd src
python preprocess_3d.py
```

**What it does:**

1. Reads every `.nii.gz` in `data/raw/Task01_BrainTumour/imagesTr/`
2. Skips macOS artifact files (prefixed `._`) and warns on missing labels
3. Extracts FLAIR (channel index 0) and T1gd (channel index 2)
4. For each channel independently:
   - Casts to `float32`, replaces NaN/Inf with 0
   - Identifies foreground voxels where intensity > 0
   - Clips to the 1st–99th percentile of those voxels (removes rare extreme outliers)
   - Min-max rescales to approximately [0, 1]
5. Binarizes the label: `mask = (label > 0).astype(uint8)`
6. Stacks channels: `np.stack([flair, t1gd], axis=0)` → shape `(2, H, W, D)` = `(2, 240, 240, 155)`
7. Saves:
   - `data/processed_3d/images/{subject_id}.npy` — float32, shape `(2, 240, 240, 155)`
   - `data/processed_3d/masks/{subject_id}.npy` — uint8, shape `(240, 240, 155)`
8. Writes `data/processed_3d/metadata.csv` with columns: `subject_id`, `image_path`, `mask_path`, `tumor_voxels`, `shape_h`, `shape_w`, `shape_d`

**Expected output:** 484 image files, 484 mask files, 1 CSV. Runs in approximately 13–15 minutes.

**Note on axis convention:** Arrays are saved in NIfTI voxel order `(C, H, W, D)` — the depth axis (D=155) is the last spatial dimension. This matches the convention used in `dataset_3d.py` and `sliding_window.py`, where patches are indexed as `image[:, h0:h0+ph, w0:w0+pw, d0:d0+pd]`.

---

### Step 2: Training

```bash
cd src
python train_3d.py
```

**Key configuration constants** (edit at top of `train_3d.py`):

| Constant | Default | Meaning |
|----------|---------|---------|
| `EXPERIMENT_NAME` | `"3d_unet_v1"` | Output directory name under `outputs/` |
| `PATCH_SIZE` | `(128, 128, 128)` | Spatial size of each training patch |
| `PATCHES_PER_SUBJECT` | `4` | Random patches sampled per subject per epoch |
| `BATCH_SIZE` | `2` | Patches per GPU forward pass |
| `NUM_WORKERS` | `2` | CPU workers for data loading (safe default; increase for faster loading) |
| `EPOCHS` | `50` | Total training epochs |
| `LR` | `1e-4` | Starting learning rate (cosine-annealed to 1e-6) |
| `GRAD_CLIP_NORM` | `1.0` | Maximum gradient norm (prevents instability early in 3D training) |
| `VAL_SW_SUBJECTS` | `10` | Subjects to run full sliding-window eval on per epoch |
| `SW_OVERLAP` | `0.5` | Fraction of patch overlap during sliding-window inference |
| `LOSS_BCE_WEIGHT` | `0.5` | Weight of BCE in combined loss |
| `LOSS_DICE_WEIGHT` | `0.5` | Weight of Dice in combined loss |

**What each epoch does:**

1. **Train phase:** for each of the 388 training subjects, sample `PATCHES_PER_SUBJECT` random 128³ patches (50% centered on a tumor voxel, 50% at random location). Pack into batches of 2, run forward pass, compute loss, backpropagate, clip gradients, update weights.
2. **Validation patch phase:** run the same patch extraction on the 96 val subjects without gradient updates. Compute validation loss and patch Dice as a fast training signal.
3. **Sliding-window validation phase:** for `VAL_SW_SUBJECTS` complete val subjects, tile 128³ inference patches across the full 240×240×155 volume with 50% overlap, average probabilities in overlapping regions, threshold at 0.5, compute full-volume 3D Dice against ground truth.
4. **Checkpoint and log:** write `training_log.csv`, save `latest_model.pt`, save `best_model.pt` if sliding-window Dice improved.

**Subject split:** the script first looks for frozen split files from `outputs/runtime_flair_t1gd_25d5_smalllesion_baseline_v1/logs/train_subjects.txt` and `val_subjects.txt`. If those exist, they are used directly to ensure the 3D run uses the **identical** 388/96 split as the 2.5D baseline. If they are absent, the script falls back to a deterministic 80/20 random split using `RANDOM_SEED=42`.

**Resuming after a crash:** the script does not currently support automatic resume. To restart from a checkpoint, modify `train_3d.py` to load `latest_model.pt` weights before the training loop and set the starting epoch accordingly.

---

### Step 3: Evaluation

```bash
cd src
python evaluate_3d.py
```

Runs **full** sliding-window inference on all 96 validation subjects (not just the per-epoch subset). For each subject:

1. Loads the full `(2, 240, 240, 155)` volume
2. Tiles 128³ patches across it with 50% overlap
3. Averages sigmoid probabilities in overlapping regions
4. Thresholds at each value in `[0.3, 0.4, 0.5, 0.6, 0.7]`
5. Computes 3D Dice vs ground truth binary mask

Outputs:
- `evaluation/subject_metrics_3d.csv` — per-subject 3D Dice at base threshold (0.5)
- `evaluation/threshold_sweep_3d.csv` — mean/median/std Dice at each threshold

**Note on checkpoints:** resolves `best_model_3d_unet_v1.pt` first; falls back to `latest_model_3d_unet_v1.pt` if best does not exist. Uses the same frozen val split as training.

---

### Step 4: Visualization

```bash
cd src
python visualize_3d.py
```

Produces three types of figures:

**1. Training curves** (`visualizations/training_curves.png`)  
Two panels: loss convergence (train + val) and Dice progression (train patch, val patch, sliding-window 3D). The best-epoch marker shows when `best_model.pt` was saved.

**2. Dice distribution** (`visualizations/dice_distribution.png`)  
Horizontal bar chart of per-subject 3D Dice sorted ascending. Bars colored red for Dice < 0.80, cyan for ≥ 0.80. Mean and median reference lines annotated.

**3. Overlay panels** (`visualizations/overlays/{best,worst,random}/`)  
For each selected subject, a 5-panel figure:
- **FLAIR:** raw grayscale MRI slice at the axial depth with the largest lesion cross-section
- **T1gd:** same slice from the contrast channel
- **GT mask:** FLAIR slice with ground-truth tumor annotated in cyan
- **Prediction:** FLAIR slice with model prediction annotated in pink
- **Probability map:** FLAIR slice with raw sigmoid probability as a hot colormap (dark = low probability, bright/white = high confidence tumor)

Subjects are categorized by their full-volume 3D Dice score:
- **best/** — top 10 subjects (title in green)
- **worst/** — bottom 10 subjects (title in red)
- **random/** — 10 randomly sampled from the middle 76 subjects (title in gold)

---

## 8. Results

### 3D U-Net — Final Evaluation (96 validation subjects)

| Metric | Value |
|--------|-------|
| Mean 3D Dice @ 0.5 | **0.894** |
| Median 3D Dice @ 0.5 | **0.923** |
| Std 3D Dice | 0.076 |
| Subjects with Dice > 0.87 | 77 / 96 (80%) |
| Worst subject (BRATS_381) | 0.555 (small tumor, ~27k voxels) |
| Best subject (BRATS_442) | 0.968 |

### Threshold Sweep

| Threshold | Mean Dice | Median Dice |
|-----------|-----------|-------------|
| 0.30 | 0.891 | 0.915 |
| 0.40 | 0.894 | 0.922 |
| **0.50** | **0.894** | **0.923** |
| 0.60 | 0.893 | 0.922 |
| 0.70 | 0.891 | 0.919 |

Threshold 0.5 is optimal. The near-flat curve indicates a well-calibrated model.

### Training Summary

- 45 of 50 epochs completed before hardware crash; model was fully converged
- Learning rate had cosine-annealed to ~3.4e-6 at epoch 45; final 5 epochs would not have changed performance
- Best sliding-window Val Dice (on 10 subjects per epoch): **0.9198** at epoch 45
- Approximate epoch time on RTX 3090: ~160 seconds
- Total training time: ~2 hours

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CPU | Intel Core i9-14700KF |
| RAM | 128 GB DDR5 |
| OS | Windows 11 |

---

## 9. Glossary of All Terms

### MRI Modalities

**FLAIR (Fluid-Attenuated Inversion Recovery)**  
An MRI sequence that nulls the signal from cerebrospinal fluid (the water that surrounds and cushions the brain). In a standard T2 scan, CSF appears bright, which can obscure lesions near the brain surface. FLAIR suppresses that signal, making tumors and edema appear bright against a dark background. It is the most diagnostically informative channel for brain tumor detection.

**T1gd (T1-weighted with Gadolinium Contrast)**  
Gadolinium is a paramagnetic contrast agent injected intravenously before scanning. It does not normally cross the blood-brain barrier (BBB). Where a tumor has broken down the BBB — as aggressive tumors do — gadolinium accumulates and causes that tissue to appear bright white on the T1-weighted scan. This reveals active, enhancing tumor tissue and helps distinguish it from surrounding edema or dead tissue.

**T1w (T1-weighted)**  
Anatomical reference scan. Fat and white matter appear bright, gray matter is medium, CSF is dark. Used for structural reference; less informative for tumor detection than FLAIR or T1gd without contrast.

**T2w (T2-weighted)**  
Water appears bright, so edema, inflammation, and CSF are all bright. Useful for detecting the extent of tumor-related swelling but less specific than FLAIR.

### Data

**NIfTI (.nii.gz)**  
Neuroimaging Informatics Technology Initiative format. A compressed binary file storing a multi-dimensional array of voxel intensities alongside a header describing physical dimensions (voxel size in mm), coordinate system orientation, and acquisition parameters.

**Voxel**  
The 3D equivalent of a pixel. One voxel represents a small cube of tissue. In MSD Task01, each voxel is approximately 1 mm × 1 mm × 1 mm. The full brain volume is 240 × 240 × 155 voxels, covering roughly the entire head.

**Ground truth mask (GT)**  
The manually annotated binary map of tumor location, produced by expert neuroradiologists drawing contours around visible tumor on each slice. Used as the training target and evaluation reference. "Binary" means all non-zero tumor subtypes (necrosis, enhancing tumor, edema) are merged: `mask = (label > 0)`.

**Binarization**  
Converting multi-class labels {0, 1, 2, 3} into a single binary class {0, 1}. Applied as `mask = (label > 0)` so the model predicts "is there any tumor here?" rather than "which type of tumor is here?". This is the standard whole-tumor segmentation task used in BraTS benchmarks.

**Axial slice**  
A 2D cross-section through the brain volume at a fixed depth (z) index. When you look at a brain scan in the axial view, you are looking down from above, like cross-sections through a loaf of bread.

### Preprocessing

**Normalization (p1/p99 min-max)**  
MRI intensity values are not calibrated — the same tissue can produce different raw numbers on different scanners or with different acquisition settings. To make all volumes comparable, we: (1) ignore background voxels with intensity = 0, (2) clip the 1st and 99th percentile of foreground intensities to remove rare extreme outliers, (3) rescale the result to [0, 1]. This is applied independently to each modality channel.

**Foreground voxels**  
Voxels with intensity > 0. In brain MRI, background voxels outside the head are zero. The normalization is computed only on foreground to avoid background noise skewing the statistics.

**Axis convention (C, H, W, D)**  
How the array dimensions are ordered in memory. C = channels (2: FLAIR and T1gd), H = height (240), W = width (240), D = depth (155). PyTorch's 3D convolutions expect input as `(Batch, Channels, D1, D2, D3)` — when data is loaded for the network it becomes `(2, 2, 128, 128, 128)` for a batch of 2 patches with 2 input channels.

### Model Architecture

**U-Net**  
A convolutional neural network architecture originally designed for biomedical image segmentation. It consists of an encoder (contracting path) that progressively downsamples the input to capture context, a bottleneck, and a decoder (expansive path) that upsamples back to the original resolution. Skip connections transfer feature maps from encoder to decoder at each scale, preserving spatial detail that would otherwise be lost. The name comes from the U-shaped diagram of the architecture.

**3D U-Net**  
A direct extension of the 2D U-Net where all `Conv2d`, `MaxPool2d`, and `ConvTranspose2d` operations are replaced with `Conv3d`, `MaxPool3d`, and `ConvTranspose3d`. This allows the network to learn spatial relationships across all three dimensions simultaneously, capturing context that is lost when treating a 3D brain as a stack of independent 2D slices.

**Encoder**  
The downsampling half of the U-Net. Each stage applies two 3×3×3 convolutions followed by a 2×2×2 max pooling to halve the spatial dimensions. Features go from 2 → 32 → 64 → 128 → 256 channels as spatial resolution decreases. Captures increasingly abstract, high-level context.

**Decoder**  
The upsampling half. Each stage uses a transposed convolution to double the spatial dimensions, then concatenates the corresponding encoder feature map (skip connection), then applies two convolutions. Combines the high-level context from deep layers with the fine spatial detail from shallow layers.

**Skip connection**  
A direct connection from an encoder layer to the corresponding decoder layer at the same spatial scale. Passes fine-grained feature maps around the bottleneck so the decoder can reconstruct precise boundaries without having to recover that detail from the compressed representation.

**Bottleneck**  
The deepest layer of the U-Net, between encoder and decoder. At this point the volume has been downsampled to 1/8 its original size (for 4 pooling layers) and the feature count is at its maximum (512 channels). Captures the most global, abstract context.

**InstanceNorm3d**  
Normalizes activations per-sample, per-channel — independently for each sample in the batch. Preferred over BatchNorm3d in 3D medical imaging because batch sizes are very small (1–2 patches) and BatchNorm's statistics become unreliable at small batch sizes. InstanceNorm has no dependence on batch size.

**LeakyReLU(0.01)**  
An activation function that passes positive values unchanged and scales negative values by 0.01 (instead of clamping them to 0 like standard ReLU). The small negative slope helps prevent "dying ReLU" neurons and is standard in 3D segmentation networks.

**Feature sizes [32, 64, 128, 256]**  
The number of feature channels at each encoder depth level. Intentionally smaller than full-scale 3D U-Nets ([64, 128, 256, 512]) to fit within 24 GB VRAM at batch size 2 with 128³ patches and mixed precision.

### Training

**Patch**  
A 128×128×128 sub-volume cropped from the full 240×240×155 brain volume. Training on the full volume is impossible due to VRAM limitations (a full float32 volume with 2 channels = 57 MB; the model activations multiply this by 10–20×). During training, the model only ever sees patches; during evaluation, sliding-window inference reassembles the full prediction.

**Foreground-biased patch sampling (50/50)**  
When randomly selecting a patch center: 50% of the time, the center is randomly chosen from voxels labeled as tumor; 50% of the time, it is completely random. Without this bias, brain tumors occupy less than 5% of brain volume on average, so most random patches contain no tumor. The model would then learn that predicting all-background gives low loss — class imbalance collapse. The 50/50 split ensures the model sees enough positive examples to learn meaningful tumor features.

**Epoch**  
One complete pass through the entire training dataset. With 388 training subjects × 4 patches per subject = 1,552 training patches per epoch, packed into batches of 2 = 776 gradient update steps per epoch.

**Batch size**  
Number of patches processed in a single forward + backward pass on the GPU. Larger batch sizes use more VRAM but enable more stable gradient estimates and often better GPU utilization. At batch size 2 with 128³ patches and AMP, peak VRAM usage was approximately 0.4 GB for the model weights (note: this is just model weights; activations during the full forward pass use considerably more).

**Loss function**  
The scalar value that the optimizer minimizes. A lower loss means better predictions. Two complementary terms are combined:

- **Binary Cross-Entropy (BCE):** treats every voxel independently. For each voxel, computes `−[y·log(p) + (1−y)·log(1−p)]` where `y` is the ground truth (0 or 1) and `p` is the predicted probability. Penalizes confident wrong predictions most heavily. Good at per-voxel accuracy.
- **Dice loss:** measures the overlap between predicted and ground-truth regions as a whole. `1 − 2·|Pred ∩ GT| / (|Pred| + |GT|)`. Explicitly penalizes size mismatches — if the model predicts a blob half the correct size, Dice loss is high even if every predicted voxel is correct. Essential for class-imbalanced segmentation.

Combined: `total_loss = 0.5 × BCE + 0.5 × Dice loss`

**Backpropagation**  
The algorithm that computes the gradient of the loss with respect to every weight in the network, working backwards from the output. These gradients tell the optimizer how to adjust each weight to reduce the loss.

**Adam optimizer**  
An adaptive gradient descent algorithm. Maintains per-parameter momentum (a running average of past gradients) and adaptive per-parameter learning rate scaling (based on gradient magnitude history). Converges faster and more reliably than vanilla stochastic gradient descent. Standard choice for medical image segmentation.

**Learning rate (LR)**  
A scalar that controls how large the weight updates are per step. Too high = unstable training (loss oscillates or diverges). Too low = very slow convergence. `1e-4` is a conservative starting point for 3D networks.

**Cosine annealing scheduler**  
Reduces the learning rate following a cosine curve from the starting value (1e-4) to a minimum (1e-6) over the total number of epochs. High LR early promotes rapid learning and coarse feature development. Low LR late enables fine-tuning without overshooting optimal weights. At epoch 45 the LR had reached ~3.4e-6 — essentially at the floor, which is why the final 5 epochs would not have produced meaningful improvement.

**Gradient clipping (`clip_grad_norm_ = 1.0`)**  
Before each weight update, if the Euclidean norm of all gradient values across the model exceeds 1.0, scales all gradients down proportionally. Prevents "exploding gradients" — a failure mode common early in 3D network training where a bad batch can cause excessively large updates that destroy accumulated learning.

**AMP (Automatic Mixed Precision)**  
Uses 16-bit floating point (float16) for most computations instead of 32-bit (float32), reducing memory by roughly half and accelerating GPU math operations. PyTorch handles automatic conversion and maintains a float32 master copy of weights to prevent precision loss during updates.

**GradScaler**  
The component of AMP that prevents float16 underflow during backpropagation. Multiplies the loss by a large scale factor before backprop (so gradients are large enough to be represented in float16), then scales the gradients back down before the optimizer step. Automatically reduces the scale factor if NaN/Inf gradients are detected.

**`persistent_workers`**  
Keeps DataLoader worker processes alive between epochs instead of spawning and killing them each epoch. Reduces overhead on long training runs at the cost of holding a small amount of RAM between epochs.

**Augmentation**  
Random transformations applied to training data to artificially increase variety and reduce overfitting. Here: random left-right, up-down, and depth-direction flips applied with 50% probability each. These are label-preserving — the mask is flipped the same way as the image.

### Evaluation

**Dice coefficient**  
The primary evaluation metric. Measures the overlap between two binary volumes:

```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```

Range: 0 (no overlap) to 1 (perfect overlap). A Dice of 0.923 means approximately 92.3% volumetric agreement between the model's predicted tumor region and the radiologist-annotated ground truth.

**3D Dice vs per-slice Dice**  
These are not the same metric. Per-slice Dice computes overlap independently on each 2D axial slice, then averages. Slices with no tumor at all contribute Dice = 1.0 (both prediction and GT are empty), inflating the average. 3D Dice computes overlap across the entire volume in a single calculation, treating all voxels equally. 3D Dice is a stricter, more clinically meaningful metric. The numbers are not comparable: a good 2D result of 0.85+ per-slice Dice is not better or worse than a 3D Dice of 0.92 — they measure different things.

**Threshold**  
The model's final layer outputs a raw logit (unbounded number). Sigmoid converts this to a probability p ∈ [0, 1]. A threshold converts the probability to a binary prediction: voxels with p ≥ threshold = tumor, others = background. At threshold 0.5 (the default): the model must be more than 50% confident to predict tumor. Lower thresholds are more sensitive (catch more tumor but have more false positives); higher thresholds are more specific (fewer false positives but may miss small lesions). The threshold sweep confirmed 0.5 is optimal for this model.

**Sliding-window inference**  
A technique for applying a patch-based model to a volume larger than the patch size. A 128³ window slides across the 240×240×155 volume with a defined stride. For each position, the model predicts a probability map for that patch. Because adjacent patches overlap by 50%, most voxels are predicted multiple times from different contexts; the final probability for each voxel is the average of all predictions that covered it. This averaging smooths out boundary artifacts at patch edges and generally produces better predictions than non-overlapping tiling.

**Val SW Dice (sw_val_dice)**  
Sliding-window 3D Dice computed on a subset of full validation volumes during training. This is the metric that determines which epoch's weights are saved as `best_model.pt`. It is monitored every epoch but only on `VAL_SW_SUBJECTS` (10) subjects for speed; the full 96-subject evaluation runs separately via `evaluate_3d.py`.

**Patch Dice (train_dice / val_dice)**  
Dice computed on individual 128³ patches, not full volumes. Includes empty background patches where both prediction and GT are all-zero — those patches contribute Dice ≈ 1.0, so this metric is optimistically biased upward. Useful as a fast training signal to confirm the model is learning, not as a clinical performance metric.

---

## 10. Understanding the Training Graphs

### Left Panel — Loss Curve

The x-axis is epoch number (1 to 45). The y-axis is the combined loss value (BCE + Dice, lower = better).

- **Red line (Train loss):** the loss computed on the training data as it is being learned. You expect this to decrease monotonically as the model improves.
- **Cyan line (Val loss):** the loss computed on held-out validation patches with no gradient updates. This tells you whether the model is generalizing or memorizing.

What healthy training looks like: both curves decrease together, with val loss slightly above train loss throughout. If val loss starts increasing while train loss keeps decreasing, that is overfitting — the model has memorized training data and stopped generalizing.

Your run shows clean convergence: both curves drop sharply in the first 10 epochs, then plateau from epoch 25 onward, with val loss tracking train loss closely. No overfitting.

### Right Panel — Dice Curve

- **Red line (Train Dice, patch):** patch-level hard Dice on training data. Inflated by empty patches; shows rapid learning in early epochs.
- **Cyan line (Val Dice, patch):** same metric on validation patches. Same inflation caveat.
- **Green dashed line (SW Val Dice, 3D):** full sliding-window volumetric Dice on 10 validation subjects. This is the real performance metric. Starts at ~0.75 in epoch 1 (the model is already somewhat correct because it learned basic anatomy fast) and rises to ~0.92 by epoch 25, then plateaus. The golden dotted vertical line marks the best epoch (45, Dice 0.9198).

The gap between patch Dice (~0.83) and SW Dice (~0.92) is expected — empty patches inflate patch Dice downward relative to full-volume Dice, while also the 50/50 sampling means training patches are artificially 50% positive whereas real brains are ~5% tumor.

---

## 11. File Reference

| File | Branch | Purpose | Run order |
|------|--------|---------|-----------|
| `preprocess_msd.py` | main | Single-channel FLAIR preprocessing (earliest version) | — |
| `preprocess_msd_flair_t1gd.py` | main | FLAIR + T1gd slice preprocessing (v1) | — |
| `preprocess_msd_flair_t1gd_25d.py` | main | 2.5D 3-slice context preprocessing | — |
| `preprocess_msd_flair_t1gd_25d5.py` | main | 2.5D 5-slice context preprocessing (current) | 1 |
| `build_runtime_metadata_flair_t1gd_25d5.py` | main | Build metadata CSV for runtime NIfTI loading | 1 (alt) |
| `dataset.py` | main | `BrainTumorSliceDataset`: runtime 5-slice 2D loading | — |
| `train.py` | main | 2.5D training loop (MONAI UNet2D) | 2 |
| `evaluate.py` | main | 2.5D evaluation with threshold sweep | 3 |
| `threshold_sweep.py` | main | Standalone threshold optimization | — |
| `plot_training_curves.py` | main | 2.5D training curve figures | 4 |
| `inspect_predictions.py` | main | Visual inspection of 2.5D failure cases | — |
| `inspect_msd.py` | main | Dataset inspection utility | — |
| `run_experiment.py` | main | Combined train + plot + eval runner | — |
| `preprocess_3d.py` | 3d-unet | 3D NIfTI → normalized .npy + metadata CSV | 1 |
| `dataset_3d.py` | 3d-unet | `BrainTumorDataset3D`: foreground-biased patch sampling | — |
| `model_3d.py` | 3d-unet | `UNet3D`: full 3D encoder-decoder with skip connections | — |
| `sliding_window.py` | 3d-unet | Overlapping patch inference for full-volume prediction | — |
| `train_3d.py` | 3d-unet | 3D training loop with SW validation | 2 |
| `evaluate_3d.py` | 3d-unet | Full-set 3D Dice evaluation + threshold sweep | 3 |
| `visualize_3d.py` | 3d-unet | Training curves, Dice distribution, best/worst/random overlays | 4 |
