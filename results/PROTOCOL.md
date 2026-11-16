# Evaluation protocol: 5-fold cross-validation (step C)

Written 2026-11-15, before any new training, on branch `protocol/cv`. It fixes how every new
number for the 3D U-Net is produced, from this date on. The April 2026 numbers (V1 0.894,
V2 0.892, ensemble 0.894 / 0.925 on 96 validation subjects) remain dated history; see
`results/AUDIT.md`. Scope stays as in the brief: the brain-tumor segmentation proof of concept
from the chemo-brain (CICI) project, self-directed, not a CICI predictor.

## 1. Why a new protocol

The audit showed the 96 April "validation" subjects chose the checkpoints (first 5 of them),
the threshold, the post-processing, TTA and the V1/V2/ensemble choice. They are not a test set,
and with 484 labeled subjects there is no clean held-out set left. Cross-validation over all
484 gives every subject exactly one prediction from a model that never saw it in training
(out-of-fold, "OOF"), and moves every decision off the scored subjects.

## 2. Data and target

- MSD Task01_BrainTumour training set, 484 labeled subjects, `data/processed_3d/`
  (2 channels: FLAIR, T1gd; 240 × 240 × 155; 1 mm isotropic).
- Target: whole tumor, `label > 0`. Subregion labels (1 edema, 2 non-enhancing, 3 enhancing) are
  used only for error analysis unless a run states otherwise.
- Known limitation: MSD gives no patient IDs, so splitting is per scan. If a patient has more
  than one scan in Task01, both could fall in different folds. Not checked yet (open item).

## 3. Folds (frozen)

- File: `results/protocol/folds_5fold_seed20261115.csv` (columns `subject_id, fold, mask_voxels,
  april_val96`), made by `src/training/make_cv_folds.py`, sha256
  `eaf1cf46c2ca972a15701a05d41910ba4703f66c9c1c3df45d66cb7b46165714`.
- 5 folds of 97, 97, 97, 97, 96 subjects. Stratified by whole-tumor volume: subjects sorted by
  mask voxel count, cut into blocks of 5, each block spread over the 5 folds in a random order
  (seed 20261115). Mask counts were re-counted from the masks and matched `metadata.csv`.
- Fold medians of tumor volume: 94.3k, 95.4k, 94.2k, 95.6k, 92.8k voxels.
- The 96 April subjects are spread over the folds (23, 20, 19, 14, 20); they get no special role.
- The folds file is never regenerated or edited. Every run reads it and records its sha256.

## 4. What each set is used for

For fold k (k = 0..4):

| Set | Subjects | Used for |
|---|---|---|
| Training | the other 4 folds (387–388) | fitting weights; nothing else |
| Test | fold k (96–97) | scored once per frozen configuration; never seen by any decision for fold k |

- **Checkpoint:** the final epoch of a fixed schedule. No early stopping, no "best" checkpoint,
  no per-epoch validation metric on the test fold. (The April V1 "best" checkpoint was its
  last epoch anyway.)
- **Training-time monitoring:** training loss and patch Dice only.

## 5. How decisions are made

1. **Inference and post-processing options** (threshold, largest-CC vs minimum component size,
   TTA, overlap, Gaussian weighting, fold ensembling of probabilities): chosen by
   **cross-fitting**. For fold k, each candidate setting is scored on the OOF predictions of
   the other four folds, the best one is picked, and only then applied to fold k. The OOF
   result reported is the combination of those per-fold choices. The candidate list is written
   in `RUNS.md` before scoring.
2. **Training changes (step E)** are compared on OOF metrics over all 484 against the current
   baseline. A change ships only if
   - the paired bootstrap 95% CI of the mean Dice difference (10,000 subject resamples,
     seed 20261115) is entirely above 0, and
   - the bottom-decile mean (mean Dice of the 48 lowest subjects) is not lower than the
     baseline's.
3. **Multiple comparisons:** every variant run is logged in `RUNS.md`, kept or not. Because
   step E compares variants on the same OOF set, the final OOF number carries a small selection
   bias; the headline will say how many variants were compared. A truly untouched estimate needs
   external data (for example later BraTS cases checked for patient overlap with Task01).
4. **Never:** dropping subjects, choosing anything on the subjects being scored, or replacing a
   dated number in place.

## 6. Metrics and reporting

Per subject (whole tumor, after the run's post-processing):

- **Dice** = 2|P∩G| / (|P|+|G|) (eps 1e-6). Empty prediction gives 0.
- **Surface Dice at 1 mm and 2 mm.** Voxel surfaces (mask minus its 6-connected erosion);
  the fraction of both surfaces within tolerance of the other.
- **HD95.** The maximum of the two directed 95th-percentile surface distances. It is undefined
  for an empty prediction; such cases are counted and listed, not silently dropped.
- **ASSD**, plus false-negative and false-positive voxel volumes split at 2 mm into boundary
  and gross errors (definitions as in `src/evaluation/error_analysis_3d.py`).

Every headline reports the following over all 484 OOF subjects:

- mean, median, IQR, SD, minimum;
- the fraction at Dice ≥ 0.80 and ≥ 0.90;
- the bootstrap 95% CI of the mean (10,000 resamples, seed 20261115);
- per-fold means and a per-case CSV.

The subset of the 96 April subjects is also shown, labeled, for continuity only. It is not
comparable one-to-one with April: different training sets, and no selection on them.

Naming: results go to `results/cv_<variant>_<date>/`; numbers are quoted as
"5-fold CV, out-of-fold, n = 484, <variant>, <date>".

The **5-model fold ensemble** (all five fold models averaged) has no honest score on these 484
(each model trained on 80% of them). Use it for demos and the website viewer only, never as a
number.

## 7. Planned runs (pre-registered)

| ID | What | Compute |
|---|---|---|
| E0 | Baseline: April V1 recipe on the 5 folds (`src/training/train_cv.py`). The table below lists the exact recipe. | ~2.3 h per fold, ~12 h total: **overnight** |
| D1 | Inference-only options on E0's saved OOF probability maps, by cross-fitting (§5.1). | minutes to ~1 h |
| E1+ | Training changes from step B evidence: all 4 channels (edema misses), spatial augmentation, longer training, region targets. Each is one full CV run. | ~12 h each: **overnight** |
| F | nnU-Net 3d_fullres on the same folds, as a reference ceiling credited to nnU-Net | about 1–2 days |

E0 recipe (identical to `src/training/train_3d.py` V1 except as noted):

- UNet3D [32, 64, 128, 256] on 2 channels.
- 128³ patches, 4 per subject per epoch, 50% centred on a tumor voxel, random flips.
- Batch 4, Adam 1e-4, cosine T_max 50 (eta_min 1e-6), 50 epochs.
- 0.5 BCE + 0.5 soft Dice, grad clip 1.0, AMP, seed 42.
- Differences from April V1:
  - training subjects come from the folds;
  - final-epoch checkpoint (April V1 was stopped after 45 of its 50 scheduled epochs and
    picked by Dice on 5 validation subjects);
  - no per-epoch validation;
  - 20 data-loader workers instead of 8 (speed only).
- E0 inference: sliding window 128³, 50% overlap, count-map averaging, AMP, no TTA,
  threshold 0.5, largest connected component (the April ensemble's settings). Results without
  largest-CC are also stored.
- E0's settings are fixed in advance; nothing is tuned on its OOF result.
