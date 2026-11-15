# HANDOFF — brain-lesion-detection U-Net investigation — 2026-11-14 — #1

Drop this file into the `brain-lesion-detection` repo (for example `docs/handoff-unet-investigation-2026-11-14.md`) and start Claude Code there, on the machine that holds `data/` and `outputs/`. The goal is better and more honest numbers for the 3D U-Net, through cleaner evaluation or tasteful changes. This brief carries what is known from outside the numbers; you have the data, checkpoints and per-case results, so do your own thinking from them and push back on anything here the numbers contradict.

## 1. Where we are

**Repo:** github.com/mattin-tasbihgoo/brain-lesion-detection (public), branch `main`, last commit `a91b6640f48305a943f48d32633019305e6f9424` ("Fuse t1 + flair imaging", Apr 19, 2026). `data/` and `outputs/` are gitignored, so the processed volumes, split files, checkpoints, logs and per-case CSVs exist only on the owner's PC (RTX 3090, 24 GB).

**What the model is:** a 3D U-Net on the Medical Segmentation Decathlon `Task01_BrainTumour` training set (484 labeled subjects). Input: 2 channels, FLAIR (channel 0) and T1gd (channel 2), normalized per channel. Target: binary whole tumor (`label > 0`). Training: random 128³ patches, BCE + Dice loss (0.5 each), Adam with cosine decay, gradient clipping 1.0, InstanceNorm. Inference: sliding window, 128³ patches, 50% overlap, plain averaging of overlaps (count map; no Gaussian weighting), threshold 0.5, then keep only the largest connected component.

- **V1:** features [32, 64, 128, 256]; 50/50 foreground-biased patches. `ENSEMBLE_ANALYSIS.md` says 45 epochs, batch 4; the README's config table says 50 epochs, batch 2. Unreconciled.
- **V2:** features [64, 128, 256, 320], deep supervision (weights 0.5, 0.25, 0.125), 70/30 foreground bias, intensity augmentation, batch 3, 50 epochs, seed 41.
- **Ensemble:** V1 and V2 probability maps averaged; TTA off.

**Split:** 388 train / 96 "validation" (80/20). Each training script first loads frozen split files from `outputs/runtime_flair_t1gd_25d5_smalllesion_baseline_v1/logs/`; if they are missing it falls back to a random split (V1 seed 42 per the README; V2 seed 41 in code). The poster says 387 / 96, which doesn't add up to 484.

**Current numbers** (`ENSEMBLE_ANALYSIS.md`, Apr 3, 2026; 96 subjects, threshold 0.5, largest-component post-processing):

| Model | Mean Dice | Median | SD |
| --- | --- | --- | --- |
| V1 | 0.894 | 0.923 | 0.076 |
| V2 | 0.892 | 0.922 | 0.095 |
| Ensemble | 0.894 | 0.925 | 0.080 |

The SBMT poster (April 2026) reports V2's numbers: mean 0.892, median 0.922, SD 0.096, $n = 96$; 85% of subjects at Dice $\geq 0.80$ and 69% at $\geq 0.90$; "best (0.96) and worst (0.74)" example panels; "state-of-the-art models on this dataset reach ~0.928"; "within 2 to 3 points of nnU-Net". The owner's site and essays use the ensemble figures, mean 0.894 and median 0.925.

Persistent failures (all three models below 0.80): BRATS_381 (26,806 voxels; V1 0.630, V2 0.237, ensemble 0.473), BRATS_096 (0.671 ensemble), BRATS_292 (149,347 voxels, 0.705), BRATS_078 (0.740), BRATS_178 (16,506 voxels, 0.761), BRATS_270 (0.777).

**How to run:** see the README: preprocess to `data/processed_3d/`, then `src/training/train_3d.py`, `src/training/train_3d_v2.py`, `src/evaluation/evaluate_3d.py` (`MODEL_VERSION` = `v1`, `v2` or `ensemble`; outputs CSVs per experiment under `outputs/`).

**Status:** the model and its numbers are from April 2026 and nothing changed after that. Known problems found while writing this brief, not yet verified on the machine: the dataset and preprocessing code is not in git (below), the evaluation subjects also chose the checkpoint and the threshold, and the poster's "worst" panel is not the worst case.

## 2. Corrections log

- **Don't call the 96 subjects an untouched test set until the audit says so.** Per the README and training code, the best checkpoint was chosen each epoch by sliding-window Dice on 10 (V1) or 5 (V2) of those same 96, the threshold sweep and the V1, V2 and ensemble choice were all judged on them, and `ENSEMBLE_ANALYSIS.md` itself calls them "validation subjects". Earlier materials call them "held-out". Say "validation" unless a later protocol creates a real test set.
- **Don't trust a clean checkout to reproduce anything.** `.gitignore` has `data/`, which also ignores `src/data/`; `train_3d_v2.py` imports `data.dataset_3d_v2`, and the README describes `dataset_3d.py` and `preprocess_3d.py`, but none of them are on GitHub. Change the pattern to `/data/` and commit the missing code (never the MRI data).
- **Confirm V1 and V2 trained on the same split before trusting the ensemble.** If the frozen split files were missing for either run, the fallback seeds differ (42 and 41), so one model may have trained on subjects the other is scored on. Check the split files each run logged.
- **Don't explain BRATS_381 by its size.** Its 26,806 voxels equal a sphere of radius about 18.6 voxels, and a 1-voxel boundary error on that sphere still scores about 0.92 (section 3). A score of 0.473 is a real miss or a label problem; `ENSEMBLE_ANALYSIS.md`'s claim that it is "a data problem" and 0.473 "the ceiling" is unverified.
- **Don't label the poster's 0.74 panel "worst".** V2's lowest case in `ENSEMBLE_ANALYSIS.md` is BRATS_381 at 0.237; 0.74 matches BRATS_078 under V2 (0.742). Find which case the poster figure shows.
- **Don't compare against nnU-Net or "~0.928" unless the protocol matches.** Published MSD and nnU-Net numbers come from a different test set, often per tumor region rather than whole tumor. The only fair comparison is nnU-Net trained and scored on the same folds here.

## 3. Constraints and settled decisions

- **Never tune on the evaluation set.** Choose checkpoints, thresholds, post-processing and every design change on training folds or an inner validation split; the reporting set is scored once per final configuration. This is the owner's rule, and it is why the current 96 are suspect.
- **No silent exclusions.** Every evaluated subject stays in the headline numbers. If a label looks wrong, document it with screenshots in a label-quality note and, at most, also report a second clearly-labeled number with that subject's reason beside it.
- **The SBMT numbers are dated history.** 0.892 on the April 2026 poster and 0.894 / 0.925 in the April analysis stay as they are; new results get new, dated names and never overwrite the old ones in docs.
- **Report the whole distribution.** Mean, median, IQR, SD, minimum, the fraction above 0.80 and 0.90, and a bootstrap 95% CI for the mean (resample subjects, 10,000 draws), plus a per-case CSV. Dice is
  $$\text{Dice}(P, G) = \frac{2\,|P \cap G|}{|P| + |G|}$$
  and is harsh on small or irregular objects. For a sphere of radius $r$ predicted one voxel short,
  $$\text{Dice} = \frac{2(r-1)^3}{(r-1)^3 + r^3},$$
  which gives about 0.68 at $r = 5$ and about 0.92 at $r = 20$. Whole-tumor masks include edema and are far less compact than spheres, so boundary disagreement costs more on them; that is why surface Dice and HD95 belong next to volumetric Dice.
- **Improving the numbers is the goal; honesty bounds it.** Mean Dice above 0.90 is a target, not a requirement. Report what is true, including null results.
- **Keep a lab notebook.** Every run gets a row in `results/RUNS.md` (date, commit, config, folds, metrics, verdict), kept or not.
- **Scope:** this is the brain-tumor segmentation proof of concept from the chemo-brain (CICI) project. It is self-directed, not USC Khan Lab work, and it is not a CICI predictor. Don't change that framing in any doc.
- **Compute:** one RTX 3090; keep runs to a few hours each and say up front which runs need an overnight slot.
- **Ask before:** deleting or overwriting checkpoints or outputs, pushing to GitHub, or posting any new number publicly (the poster, site and essays use the old ones).

## 4. Rejected, and why not to revisit

- Dropping hard cases (BRATS_381, BRATS_292, and the rest) from the reported set — that improves the number by hiding the failure; label concerns get documented, not deleted.
- Picking checkpoints, thresholds or post-processing on the reporting set — that is the leak this investigation exists to remove.
- Chasing 0.90 without error analysis first — blind tuning wastes GPU time and finds noise.
- Reporting an nnU-Net run as the owner's model — it is a reference ceiling, credited as nnU-Net.
- Mixing old and new numbers in one table without dates — the SBMT figures are history.

## 5. Open questions

- What does "0.894" measure, exactly? Whole tumor only, at threshold 0.5, with largest-component post-processing, on 96 subjects also used for selection. The audit answers this in writing.
- Which number was on the printed poster (V2's 0.892) and which belongs on the site (the ensemble's 0.894, or a new cross-validated number)? The owner decides after the audit; the site handoff waits on it.
- Is largest-component post-processing hurting multifocal tumors? Compare per case with and without it, and against a minimum-component-size rule chosen on training folds.
- Is BRATS_381 a model miss or a label problem? The same question for BRATS_292 (large tumor, about 0.70 everywhere).
- Does the 2-channel choice still hold in 3D? FLAIR + T1gd won an ablation in the 2D pipeline; T1 and T2 are unused.
- Would a real test set be possible? Options: 5-fold cross-validation over all 484 subjects (out-of-fold numbers for every case, with selection inside each fold), or an external set such as a later BraTS release; MSD Task01 comes from older BraTS data, so check for overlapping patients before using any BraTS release.
- Should the investigation adopt nnU-Net as the reference ceiling on the same folds? Likely yes, as a comparison only.
- License and citations for anything public: confirm the MSD data license and cite Antonelli et al., Nature Communications, 2022 (and the BraTS papers it asks for) wherever images or numbers appear, including the website's slice viewer.
- Is the owner's sense that the "worst" case "was still spot on" right? Surface Dice at a 1 to 2 mm tolerance answers it: high surface Dice with low volumetric Dice means boundary disagreement, not a miss.

**Suggested order after the audit** (adjust from what the audit finds):
- **B, error analysis:** Dice against tumor volume, surface Dice at 1 and 2 mm, HD95, false-positive and false-negative volume per case, the effect of largest-component post-processing per case, and visual review (FLAIR, T1gd, label, prediction) of the six persistent failures.
- **C, honest protocol:** write `results/PROTOCOL.md` (folds, what each set is used for, metrics) before any new training.
- **D, inference-only gains**, judged on folds: Gaussian-weighted sliding window, flip TTA, more overlap, a size-threshold post-processing rule, ensembling fold models.
- **E, training changes:** 5-fold CV (also yields a 5-model ensemble), all four channels, longer training (50 epochs of 4 patches per subject is short), spatial augmentation (rotation, scaling, elastic, gamma), mixed precision, and optionally region-based targets (whole tumor, tumor core, enhancing tumor) for per-region numbers.
- **F, reference:** nnU-Net (3d_fullres) on the same folds as a ceiling.
- **G, website assets:** the final per-case table and best, median and worst axial slices with prediction and label outlines and Dice under each, for the site's slice viewer.

A change ships only if its cross-validated mean beats the baseline by more than the bootstrap noise and doesn't make the worst decile worse.

## 6. Next action

Run the audit and write `results/AUDIT.md` on a new branch: fix the `.gitignore` pattern and commit the missing `src/data/` code; confirm both runs used the same frozen split, that 388 + 96 = 484 with no subject in both lists, and which subjects chose the checkpoints; re-run `evaluate_3d.py` for V1, V2 and the ensemble to regenerate per-case CSVs; and state in writing what the 96 subjects were used for and which case the poster's "worst" panel shows.

**Done when:** the regenerated ensemble per-case table reproduces mean 0.894 and median 0.925 to within 0.001 (or `AUDIT.md` explains the gap), the split is shown leak-free or the leak is documented, and `AUDIT.md` is committed on the branch with no training run started.
