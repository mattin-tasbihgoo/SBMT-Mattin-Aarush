# Error analysis (step B) — 2026-11-15

Scope: handoff step B, for the April 2026 V1, V2 and ensemble checkpoints on the 96 frozen **validation** subjects. This is analysis only. **Nothing here chooses a configuration.** These 96 already made the April selection decisions (`results/AUDIT.md` §5), so every suggestion in §6 must be tested on cross-validation folds before it is adopted. No training was run.

- Script: `src/evaluation/error_analysis_3d.py`, with stages `infer` (GPU, about 9 min), `metrics` (20 CPU workers, 46 s), `plots`, `figures`.
- Probability maps: `outputs/error_analysis_2026-11-15/probs/` (float32, 6.4 GB).
- Per-case table: `results/error_analysis_2026-11-15/per_case_metrics.csv`. It has one row per subject × model × post-processing (`cc` = keep the largest connected component, the April headline; `nocc` = threshold only).
- Summary with bootstrap CIs: `results/error_analysis_2026-11-15/summary.csv`.
- Plots: `summary_plots.png`, `error_volumes.png`. These contain no patient images.
- Per-case MRI review panels for 17 cases: `outputs/error_analysis_2026-11-15/case_figures/` (local only, not committed; see §8).
- Self-check: Dice from the saved maps reproduces `results/audit_2026-11-15` exactly for all 3 models × 2 post-processing variants × 96 cases.

## 1. Metrics and how to read them

The voxel spacing is 1 mm isotropic, from the NIfTI headers. Processed masks equal raw `label > 0` voxel for voxel. Subregion labels: 1 edema, 2 non-enhancing tumor, 3 enhancing tumor.

- **Surface Dice @ t mm:** the share of both boundaries lying within t mm of the other boundary. It is a voxel-count approximation of Nikolov et al. 2021, not area-weighted. Sanity check on synthetic spheres: a prediction one voxel short gives Dice 0.67 at r = 5 and 0.92 at r = 20, but **surface Dice @2 mm = 1.00 in both cases**. At 1 mm the same 1-voxel error still scores only about 0.8, because diagonal voxel neighbours are √2 mm apart. **So 2 mm is the tolerance that separates boundary noise from real errors; 1 mm partly penalises 1-voxel disagreement.**
- **HD95:** the larger of the two directed 95th-percentile boundary distances (MONAI convention), in mm.
- **Boundary vs gross error:** a missed voxel (FN) is "boundary" if it lies within 2 mm of the prediction, otherwise "gross". A wrongly predicted voxel (FP) is "boundary" if it lies within 2 mm of the label, otherwise "gross".

## 2. Headline distribution (ensemble, thr 0.5, n = 96)

| Post-processing | Metric | Mean [95 % CI] | Median | IQR | Min |
|---|---|---|---|---|---|
| largest CC (April headline) | Dice | 0.894 [0.877, 0.909] | 0.925 | 0.872 – 0.946 | 0.473 |
| | surface Dice @1 mm | 0.742 [0.710, 0.772] | 0.786 | 0.667 – 0.845 | 0.121 |
| | surface Dice @2 mm | 0.858 [0.829, 0.883] | 0.909 | 0.803 – 0.946 | 0.238 |
| | HD95 (mm) | 10.3 [7.7, 13.1] | 3.5 | 2.4 – 11.4 | max 58.8 |
| | ASSD (mm) | 1.90 [1.50, 2.37] | 1.07 | 0.79 – 2.08 | max 14.1 |
| no CC | Dice | 0.898 [0.881, 0.912] | 0.926 | 0.888 – 0.946 | 0.471 |
| | surface Dice @2 mm | 0.864 [0.834, 0.889] | 0.912 | 0.813 – 0.944 | 0.237 |
| | HD95 (mm) | 8.6 [6.4, 11.2] | 3.5 | 2.4 – 9.2 | max 65.5 |

V1 and V2 alone look much the same; the full table is in `summary.csv`.

- **The typical case is at boundary level:** 51 of 96 cases have surface Dice @2 mm ≥ 0.90 and 72 of 96 have ≥ 0.80. The median HD95 is 3.5 mm.
- **The mean HD95 (10 mm) comes from a tail** of 14 cases with HD95 > 20 mm (30 cases > 10 mm), all with errors far from the tumor.

## 3. Dice against tumor volume

| GT volume tertile | Range (cm³) | Mean Dice | Median Dice | Median sDice@2mm | Median HD95 | Cases < 0.80 |
|---|---|---|---|---|---|---|
| small | 10 – 67 | 0.878 | 0.906 | 0.918 | 3.3 mm | 6 / 32 |
| mid | 67 – 113 | 0.878 | 0.899 | 0.830 | 8.3 mm | 8 / 32 |
| large | 116 – 264 | 0.925 | 0.942 | 0.924 | 3.2 mm | 2 / 32 |

The Spearman correlation between volume and Dice is ρ = 0.38 (p = 1e-4), so it is real but weak. Small tumors lose some Dice, as the sphere formula predicts, but their surface agreement is as good as that of large ones. **Mid-size tumors have the worst surface metrics.** Size explains part of the Dice spread; it does not explain the failures. BRATS_381 (27 cm³) is a real miss (§4), not a size artefact, as the handoff predicted.

## 4. The low cases: what kind of error each one is

Review set: every case with ensemble Dice < 0.80, plus any case where largest-CC changes Dice by more than 0.05 (17 cases). Classification rules, all relative to GT volume:
- CC deleted true tumor: the step removed true-positive voxels > 5 % of GT.
- Gross over-segmentation: gross FP > 10 % and larger than gross FN.
- Gross under-segmentation: gross FN > 10 %, at least gross FP, and not mainly caused by CC.
- Boundary-level: surface Dice @2 mm ≥ 0.80.

**Viewed** means I looked at the review panel; the other rows are classified from the metrics only.

| Case | Dice CC / no CC | sDice@2 | Gross FN / FP (share of GT) | Edema missed | Type | Viewed |
|---|---|---|---|---|---|---|
| BRATS_381 | 0.473 / 0.470 | 0.24 | 0.50 / 0.00 | 83 % | **Under-segmentation of faint edema** | ✔ |
| BRATS_096 | 0.671 / 0.671 | 0.29 | 0.00 / 0.65 | 0 % | **Over-segmentation (halo + temporal spread)**; label review candidate | ✔ |
| BRATS_292 | 0.705 / **0.912** | 0.59 | 0.39 / 0.01 | 38 % | **Largest-CC deleted a separate, correctly found lesion** | ✔ |
| BRATS_078 | 0.740 / 0.737 | 0.51 | 0.27 / 0.06 | 37 % | **Missed region is cerebellar "edema"**; label review candidate | ✔ |
| BRATS_178 | 0.761 / 0.760 | 0.82 | 0.15 / 0.08 | 21 % | Mixed, mostly boundary-level | |
| BRATS_270 | 0.777 / 0.777 | 0.69 | 0.19 / 0.01 | 44 % | Under-segmentation (edema) | |
| BRATS_224 | 0.777 / 0.777 | 0.66 | 0.21 / 0.00 | 42 % | Under-segmentation (edema) | |
| BRATS_197 | 0.786 / 0.792 | 0.68 | 0.21 / 0.01 | 28 % | Under-segmentation; CC also removed some tumor | |
| BRATS_318 | 0.786 / **0.881** | 0.78 | 0.28 / 0.00 | 37 % | **Largest-CC deleted a correctly found second focus** | ✔ |
| BRATS_376 | 0.788 / 0.798 | 0.76 | 0.17 / 0.01 | 57 % | Under-segmentation (edema) | |
| BRATS_154 | 0.789 / **0.937** | 0.76 | 0.28 / 0.00 | 46 % | Largest-CC deleted true tumor (GT 74 % in its largest component) | |
| BRATS_075 | 0.790 / 0.790 | 0.64 | 0.00 / 0.34 | 2 % | Over-segmentation (anterior extension); ambiguous | ✔ |
| BRATS_315 | 0.791 / 0.791 | 0.76 | 0.08 / 0.06 | 21 % | Mixed, small | |
| BRATS_366 | 0.795 / 0.795 | 0.69 | 0.03 / 0.18 | 17 % | Over-segmentation | |
| BRATS_179 | 0.797 / 0.784 | 0.84 | 0.10 / 0.07 | 15 % | Mostly boundary-level; CC helped | |
| BRATS_430 | 0.799 / 0.799 | 0.76 | 0.11 / 0.05 | 22 % | Under-segmentation (edema) | |
| BRATS_101 | 0.801 / 0.719 | 0.63 | 0.05 / 0.18 | 5 % | Over-segmentation; **CC helped** (+0.08, removed 23.5k FP voxels) | |

Visual notes. These are observations from the panels, not a radiological read:

- **BRATS_381.** The label covers a faint, diffuse FLAIR hyperintensity in the temporal lobe, with almost no enhancement (406 enhancing voxels). The model segments the brighter core and misses a thick outer rim. The label looks plausible, so this is a **model miss on low-contrast edema**. FLAIR and T2 look blocky in the coronal and sagittal views, which suggests thick-slice acquisitions. The April claim that 0.473 is "the ceiling" and "a data problem" is not supported.
- **BRATS_096.** The model misses only 54 voxels but predicts a 5–10 mm halo around the mass and spreads into the temporal lobe. The label is mostly "enhancing" (64 %) with a thin edema rim, and FLAIR shows hyperintensity beyond the label on inferior slices. **Possible under-labelled edema; needs review.** The T1 volume looks unlike other subjects' T1 (the model does not use T1). There is also a small detached labelled fragment near the orbit (axial 27).
- **BRATS_292.** There are two separate lesions: a large temporal one (mostly edema) and a separate frontal ring-enhancing one. The model finds both. Largest-CC deletes the frontal lesion, about 49k correctly predicted voxels. **Not a model or label failure: a post-processing failure.** The images are low resolution and blocky.
- **BRATS_078.** The main temporal lesion is segmented well. Nearly all gross misses are in the **cerebellum**, where the label marks bilateral patchy "edema" following the folia, and I could not see a matching FLAIR hyperintensity on the reviewed slices. **Strongest label-quality candidate in the set; needs expert review.**
- **BRATS_075.** The prediction extends anteriorly in the temporal and frontal lobe beyond the label. FLAIR is equivocal there, so this could be over-calling or under-labelling.
- **BRATS_318.** There are two enhancing foci, parietal and temporal, joined in the label by a thin edema bridge, so GT is one component. The prediction misses the bridge, splits in two, and largest-CC deletes the correctly found temporal focus.

## 5. Patterns across all 96

- **Edema is what gets missed.** Pooled over all 96 (ensemble, CC), the missed share is 12.7 % for edema, 7.8 % for non-enhancing tumor and 2.6 % for enhancing tumor; per-case medians are 11.6 %, 1.3 % and 0.4 %. In the under-segmentation cases of §4, edema makes up 78–100 % of the missed voxels. The exception is BRATS_197, at 55 % edema and 44 % non-enhancing tumor.
- **FP volume is mostly near the boundary; FN volume is not.** 74 % of all FP voxels lie within 2 mm of GT, but only 53 % of FN voxels lie within 2 mm of the prediction. In a typical case (median), 24 % of missed voxels and 5 % of false-positive voxels are gross. So the model mostly over-reaches slightly at boundaries and occasionally misses whole regions.
- **Largest-CC is net negative on these 96 and hits a specific failure mode.** It costs more than 0.01 Dice on 5 cases (ensemble) and gains more than 0.01 on 3. The losses (BRATS_292 −0.21, BRATS_154 −0.15, BRATS_318 −0.09) are **multifocal tumors where the model was right**. The gains (BRATS_101 +0.08, BRATS_090 +0.03) come from removing spurious blobs. 9 of 96 labels have less than 95 % of their volume in one connected component.
- **Over-segmentation cases (096, 075, 366, 101) are where label and model disagree about the edema boundary.** Some may be label issues; that needs expert review, not a metric.

## 6. Questions from the handoff, answered

| Question (handoff §5) | Answer from these data |
|---|---|
| Is largest-CC hurting multifocal tumors? | **Yes.** It deletes correctly segmented second lesions in BRATS_292, BRATS_154 and BRATS_318 (−0.09 to −0.21 Dice each) and costs 0.004 mean Dice overall (CI crosses 0). Deciding what to use instead (none, or a minimum-component-size rule) is step D, **on folds**. |
| Is BRATS_381 a model miss or a label problem? | **Model miss**, most likely: under-segmentation of faint edema with a plausible-looking label. Surface Dice @2 mm is 0.24, so it is not boundary disagreement. |
| Is BRATS_292 a model miss or a label problem? | **Neither: post-processing.** Without largest-CC it scores 0.91. |
| Was the poster's "worst" case (BRATS_178, V1 no-CC, Dice 0.74) "still spot on"? | **Mostly but not fully.** Surface Dice is 0.81 @2 mm and 0.70 @1 mm, which ranks 26th lowest of 96 (median 0.91). Most of the boundary is within 2 mm, but about 11 % of the tumor is grossly missed and about 12 % grossly over-called. HD95 is 29.5 mm. |
| Does the 2-channel choice still hold? | Not tested here. The edema finding makes **T2**, which is unused and edema-sensitive, the obvious channel to test in step E. |

## 7. What this suggests for later steps (hypotheses to test on folds, not conclusions)

1. **Step D:** compare largest-CC with no CC and with a minimum-component-size rule chosen inside each training fold. Watch BRATS_101-type cases, where CC helps.
2. **Step E:** target edema under-segmentation. Options include adding T2 (and T1) channels, longer training, spatial and gamma augmentation, and region-based outputs, so that whole-tumor errors can be traced to subregions.
3. **Label-quality note:** have BRATS_078 (cerebellar "edema"), BRATS_096 (edema rim) and BRATS_075 reviewed by someone qualified. Per the protocol they stay in every headline number regardless; at most they get a second, clearly labelled number.
4. **Acquisition quality:** BRATS_381, 292 and 078 have visibly low through-plane resolution. This could be quantified as a covariate later.

## 8. Not done

- No training, and no selection of thresholds or post-processing.
- Six of 17 review cases were inspected visually; the rest were classified from metrics only. Their panels are in `outputs/error_analysis_2026-11-15/case_figures/`.
- The MRI review panels are **not committed**. MSD Task01 is CC-BY-SA 4.0, so publishing images needs attribution (Antonelli et al., Nature Communications 2022, plus the BraTS papers) and share-alike. Committing them is the owner's call.
- No radiologist review: all label comments are flags, not verdicts.
