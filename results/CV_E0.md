# E0: 5-fold cross-validation baseline (April V1 recipe), 2026-11-15/16

Protocol: `results/PROTOCOL.md` (folds file sha256 `eaf1cf46…6165714`). Code commit `4f08374`
(branch `protocol/cv`). All configs, training logs and per-fold CSVs are in
`results/cv_e0_2026-11-15/fold_logs/`. The per-case out-of-fold (OOF) table for all 484
subjects is `results/cv_e0_2026-11-15/per_case_oof.csv`, and the summary is `summary.csv`.

Quote as: **"5-fold CV, out-of-fold, n = 484, E0 (V1 recipe), 2026-11-16"**.

## 1. What was run

The April V1 recipe was trained 5 times, once per fold:

- UNet3D [32–256], FLAIR + T1gd, 128³ patches, 50 epochs, Adam 1e-4 with cosine decay,
  BCE + Dice, AMP, seed 42.
- Each fold model trained on 387–388 subjects and scored only its own 96–97 test subjects.
- The checkpoint is the final epoch. Nothing was selected on test subjects, and no test metric
  existed until training ended.
- Inference was fixed in advance: sliding window, 50% overlap, no TTA, threshold 0.5, largest
  connected component (CC).

Run details:

- **Wall time:** 22:44 → 08:58, about 2 h per fold (about 142 s per epoch, GPU-bound).
- **Crashed attempt:** an attempt at 22:27 crashed in epoch 1, when data-loader workers hit
  the Windows commit limit. It was restarted from scratch with 14 workers; no fold resumed
  mid-run.
- **Training curves:** final training losses were 0.100–0.105 on all 5 folds, close to April
  V1's 0.105 at its last epoch (45).

## 2. Headline (pre-registered E0 configuration: threshold 0.5 + largest CC)

| Set | n | Mean | 95% CI (bootstrap) | Median | IQR | SD | Min | ≥ 0.80 | ≥ 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| All, OOF | 484 | **0.8972** | 0.8908–0.9035 | 0.9205 | 0.8786–0.9449 | 0.0717 | 0.521 | 89.5% | 66.1% |
| April's 96 (subset, for continuity) | 96 | 0.8935 | 0.8775–0.9082 | 0.9197 | 0.8777–0.9471 | 0.0772 | 0.521 | 85.4% | 63.5% |

- **Per-fold mean Dice:** 0.8940, 0.9073, 0.8831, 0.9073, 0.8943. Fold medians are all
  0.919–0.922.
- **Bottom decile** (mean of the 48 lowest): 0.7242. This is the baseline for the ship rule.
- **Surface metrics (all 484):**
  - surface Dice @1 mm: median 0.758 (mean 0.732);
  - surface Dice @2 mm: median 0.892 (mean 0.859);
  - HD95: median 3.9 mm (mean 9.7 mm); 58 cases are above 20 mm;
  - ASSD: median 1.1 mm.
- **Dice by tumor-volume tertile:** small 0.864, mid 0.909, large 0.920.
- **Missed-voxel share by subregion:** edema 13.0%, non-enhancing 6.3%, enhancing 3.3%.
  Edema remains the main miss, as in step B.
- **Precision check:** Dice recomputed from the stored float16 probability maps agrees with
  the float32 scoring to within 7e-4 per case (5 of 484 differ by more than 1e-4). The mean is
  identical to 5 decimals.

## 3. What this says about the April numbers

- **The selection leak was small.** On the same 96 subjects, a V1-recipe model with **no**
  selection on them scores 0.8935. The April numbers were V1 0.8915 (no TTA, same settings),
  V1 0.8928 (TTA) and the ensemble 0.8939. Choosing the checkpoint, threshold and post-processing
  on those subjects did not measurably inflate the April numbers. The models differ (different
  training sets, one model vs ensemble), so this is consistent evidence, not an exact
  decomposition.
- **The 96 are a slightly harder-than-average draw.** They score 0.8935 here against 0.8972 for
  all 484. The honest whole-dataset estimate of this recipe is 0.897 (CI 0.891–0.904).
- **Persistent failures stay low out-of-fold:**

  | Subject | Dice (CC) | Note |
  |---|---|---|
  | BRATS_381 | 0.521 | the lowest of all 484 |
  | BRATS_096 | 0.699 | |
  | BRATS_292 | 0.707 | 0.911 without CC |
  | BRATS_078 | 0.735 | |
  | BRATS_178 | 0.772 | |
  | BRATS_270 | 0.713 | |

## 4. New evidence for step D (not acted on; decisions go through cross-fitting)

Largest-CC hurts more cases than it helps here:

- Without it, OOF mean Dice is 0.9015 instead of 0.8972 (median 0.9213), with 91.5% of cases
  at ≥ 0.80.
- 41 cases lose more than 0.005 Dice to CC and 31 gain more than 0.005.
- Several new worst cases are CC deletions of correctly found tumor:

  | Subject | Dice with CC | Dice without CC |
  |---|---|---|
  | BRATS_258 | 0.635 | 0.892 |
  | BRATS_123 | 0.669 | 0.888 |
  | BRATS_125 | 0.674 | 0.923 |
  | BRATS_364 | 0.680 | 0.918 |
  | BRATS_066 | 0.699 | 0.913 |
  | BRATS_154 | 0.786 | 0.935 |

Per `PROTOCOL.md` §5.1, switching post-processing is decided by cross-fitting over the folds
(run D1), not by reading this table. The no-CC number above is **not** a result of a frozen
configuration and must not be quoted as one.

## 5. Not done / caveats

- E0's numbers are for whole tumor only, one model per subject, with no TTA.
- The 5-model fold ensemble was not scored and cannot be honestly scored on these 484
  (`PROTOCOL.md` §6).
- Patient-level duplicates in MSD Task01 are still unchecked (`PROTOCOL.md` §2).
- Nothing here is comparable with published BraTS/MSD leaderboard numbers.
