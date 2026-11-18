# E1: four input channels, 5-fold CV (2026-11-17)

The E0 recipe with 4 input channels (FLAIR, T1gd, T1w, T2w) instead of 2 (FLAIR, T1gd), scored
with the frozen D1 inference configuration. Pre-registered in `results/RUNS.md` (commit
`defc881`) before training. Code: `src/training/train_cv.py --recipe E1`,
`src/evaluation/score_fold_e.py`, `src/evaluation/compare_cv.py`. Files are in
`results/cv_e1_2026-11-17/`:

- `per_case_oof.csv`: all metrics per subject, with the E0 + D1 baseline columns (`*_base`) and
  E1 under E0's inference configuration (`dice_e0cfg`);
- `summary.csv`: all 484, April's 96 subset and each fold, for Dice, surface Dice @1/2 mm, HD95
  and ASSD;
- `ship_rule.csv`;
- `fold_logs/`: training logs, training and scoring configs (with checkpoint sha256), per-fold
  scores, and `fold3_resumed.txt`.

Quote as: **"5-fold CV, out-of-fold, n = 484, E1 (4-channel) + D1 inference, 2026-11-17"**.

## 1. What changed

- **Changed:** input channels, 2 → 4. The images come from `data/processed_3d_4ch`, with the same
  per-channel normalization, stored as float16. Channels 0–1 were verified identical to E0's
  inputs.
- **Unchanged:** folds (`folds_5fold_seed20261115.csv`), model widths [32, 64, 128, 256], 128³
  patches, 50/50 foreground sampling, flips, batch 4, 50 epochs, cosine schedule, seed 42 and the
  final-epoch checkpoint.
- **Inference fixed in advance** (D1): flip TTA, count averaging, 50% overlap, threshold 0.5,
  keep every component at least 10% of the largest. Nothing was re-tuned on E1's out-of-fold
  results.
- **Hypothesis (from step B and E0):** edema is the main miss. T2w shows edema directly, so adding
  it should reduce edema misses.

## 2. Result and ship rule

| | n | Mean | 95% CI | Median | IQR | SD | Min | ≥ 0.80 | ≥ 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| **E1 + D1, OOF** | 484 | **0.9092** | 0.9035–0.9146 | **0.9280** | 0.8897–0.9486 | 0.0619 | 0.525 | 93.2% | 71.1% |
| E0 + D1 (baseline), OOF | 484 | 0.9036 | 0.8979–0.9091 | 0.9218 | 0.8844–0.9452 | 0.0628 | 0.546 | 92.1% | 68.4% |
| E1 + D1, April's 96 subset | 96 | 0.9060 | 0.8917–0.9188 | 0.9298 | 0.8821–0.9492 | 0.0688 | 0.525 | 90.6% | 67.7% |
| E0 + D1, April's 96 subset | 96 | 0.8987 | 0.8835–0.9124 | 0.9221 | 0.8813–0.9466 | 0.0723 | 0.546 | 88.5% | 65.6% |

**Ship rule (PROTOCOL.md §5.2) against E0 + D1: passes.**

- Paired mean difference: **+0.0056** (95% CI +0.0036 to +0.0075; 10,000 resamples, seed
  20261115).
- Bottom-decile mean (the 48 lowest cases): **0.7621**, against the required ≥ 0.7542.
- Per case: 219 subjects improve by more than 0.005 and 74 get worse by more than 0.005. 14
  improve by more than 0.05 and 7 get worse by more than 0.05.

Every fold improves:

| Fold | n | E0 + D1 | E1 + D1 | Change | Better / worse by > 0.005 |
|---|---|---|---|---|---|
| 0 | 97 | 0.9034 | 0.9074 | +0.0040 | 43 / 16 |
| 1 | 97 | 0.9141 | 0.9168 | +0.0027 | 37 / 17 |
| 2 | 97 | 0.8946 | 0.9018 | +0.0073 | 51 / 8 |
| 3 | 97 | 0.9055 | 0.9131 | +0.0076 | 41 / 16 |
| 4 | 96 | 0.9005 | 0.9068 | +0.0062 | 47 / 17 |

Surface metrics (all 484):

| | E1 + D1 | E0 + D1 |
|---|---|---|
| Surface Dice @2 mm, median (mean) | 0.907 (0.879) | 0.898 (0.868) |
| Surface Dice @1 mm, median (mean) | 0.789 (0.761) | 0.764 (0.740) |
| HD95, median (mean) | 3.6 mm (7.8 mm) | 3.7 mm (8.4 mm) |
| ASSD, median | 1.02 mm | 1.07 mm |
| Cases with HD95 > 20 mm | 43 | 44 |

No prediction was empty. 29 cases keep more than one component, the same as the baseline.

## 3. Did it do what the hypothesis said? (descriptive; not used for any choice)

**Mostly yes, but the effect is modest.** The gain comes mainly from fewer missed voxels, and
most of all in the cases where the baseline missed the most edema.

| Missed-voxel share (all 484, pooled) | E1 + D1 | E0 + D1 |
|---|---|---|
| Edema | **10.6%** | 11.9% |
| Non-enhancing core | 4.4% | 5.0% |
| Enhancing tumor | 2.0% | 2.1% |

- Edema misses fall by about a tenth (11.9% → 10.6%), and 317 of 484 cases miss fewer edema
  voxels. Edema is still by far the largest miss. The pre-registration quoted 13%, which was
  under E0's inference; under the D1 baseline the share is 11.9%.
- Pooled over all cases, missed voxels fall 10.6% and false-positive voxels fall 2.3%. The median
  predicted-to-true volume ratio moves from 0.990 to 0.996.
- The gain grows with how much edema the baseline missed:

  | Quartile of baseline edema-miss fraction | n | Mean change |
  |---|---|---|
  | Q1 (least missed) | 121 | +0.0010 |
  | Q2 | 121 | +0.0033 |
  | Q3 | 121 | +0.0046 |
  | Q4 (most missed) | 121 | +0.0132 |

- **The gain does not depend on D1's inference.** Under E0's inference configuration (no TTA,
  largest component), E1 scores 0.9022 against E0's 0.8972: +0.0050 (CI +0.0028 to +0.0070). The
  two gains roughly add up: E0 → E0 + D1 is +0.0064, and E0 → E1 + D1 is +0.0120.

## 4. Per-case changes

The biggest gains are mostly under-segmented cases that recover missed edema:

| Subject | E0 + D1 | E1 + D1 | Change | Missed voxels, baseline → E1 |
|---|---|---|---|---|
| BRATS_223 | 0.781 | 0.909 | +0.128 | 89,449 → 26,538 |
| BRATS_134 | 0.749 | 0.852 | +0.103 | 22,222 → 12,528 |
| BRATS_400 | 0.788 | 0.883 | +0.095 | 7,455 → 1,435 (false positives 58,610 → 32,786) |
| BRATS_318 | 0.782 | 0.875 | +0.093 | 32,045 → 11,525 |
| BRATS_414 | 0.726 | 0.810 | +0.085 | 67,165 → 49,323 |

The biggest losses are mostly the opposite: E1 misses more of the tumor.

| Subject | E0 + D1 | E1 + D1 | Change | What changed |
|---|---|---|---|---|
| BRATS_114 | 0.932 | 0.743 | −0.189 | missed voxels 6,966 → 29,444 (edema 5,969 → 17,167) |
| BRATS_189 | 0.823 | 0.702 | −0.121 | false positives 11,708 → 27,882 |
| BRATS_368 | 0.879 | 0.766 | −0.113 | missed voxels 10,470 → 28,069 |
| BRATS_363 | 0.810 | 0.749 | −0.061 | missed voxels 46,984 → 61,734 |
| BRATS_062 | 0.687 | 0.629 | −0.059 | missed voxels 24,080 → 29,716 |
| BRATS_352 | 0.789 | 0.732 | −0.058 | missed voxels 22,358 → 34,304 |
| BRATS_444 | 0.856 | 0.804 | −0.052 | missed voxels 10,344 → 15,422 |

E1 under E0's inference shows the same losses (for example BRATS_114 at 0.729), so they come from
the model, not from the inference configuration. They are listed for the next error review, not
tuned on.

**Persistent failures.** Four of the five improve a little. BRATS_381 does not move and is still
the lowest case. Its missed edema is unchanged (14,535 of 19,290 voxels against 14,830), so adding
T2w did not help with its faint, diffuse edema.

| Subject | E0 | E0 + D1 | E1 + D1 |
|---|---|---|---|
| BRATS_381 | 0.521 | 0.546 | 0.525 |
| BRATS_096 | 0.699 | 0.698 | 0.729 |
| BRATS_270 | 0.713 | 0.704 | 0.737 |
| BRATS_078 | 0.735 | 0.734 | 0.780 |
| BRATS_178 | 0.772 | 0.773 | 0.788 |

**Cases D1 made worse:**

| Subject | E0 | E0 + D1 | E1 + D1 | Note |
|---|---|---|---|---|
| BRATS_477 | 0.921 | 0.802 | 0.815 | A false second region is still kept by the 10% rule (E1 under E0's inference: 0.919) |
| BRATS_132 | 0.901 | 0.845 | 0.913 | Now a single component |
| BRATS_428 | 0.864 | 0.813 | 0.889 | Now a single component |
| BRATS_168 | 0.867 | 0.821 | 0.821 | Still 2 components (E1 under E0's inference: 0.857) |

**The new 10 lowest cases:** BRATS_381 0.525, 062 0.629, 340 0.636, 138 0.669, 077 0.689, 106
0.695, 189 0.702, 452 0.711, 299 0.719 and 169 0.722.

## 5. Run notes

- **Timing:** 2026-11-17 10:51 → 23:40 on the RTX 3090, with 14 workers. Each fold trained in
  2.0–2.2 h, the same as E0. The input data is 33 GB of float16 on `D:`.
- **Fold 3 was paused and resumed.** It was stopped by hand after epoch 20 (17:57) and resumed at
  20:05 (`fold_logs/fold3_resumed.txt`).
  - The first resume attempt crashed: the saved CPU random-number state was loaded onto the GPU,
    and `torch.set_rng_state` rejects that. `run_cv.py` retried four times and gave up; the
    tracebacks are in `outputs/cv_e1_4ch_2026-11-17/run.log`.
  - The fix (commit `a4b6221`) is one line and leaves the recipe unchanged. The run was then
    restarted; folds 0–2 were skipped and fold 3 continued from epoch 21.
  - **A resumed fold is not bit-identical to an uninterrupted one.** The data-loader workers
    restart with fresh seeds, so the patch and flip draws after epoch 20 differ. The training loss
    shows no break: it fell 0.011 from epoch 20 to 21, against 0.008–0.009 in the other E1 folds.
    Fold 3 is in line with the other folds (+0.0076, the largest gain, close to fold 2's +0.0073).
- **Code versions:**
  - Fold 0 trained at `defc881`; folds 1–3 at `31da66c`, which only added a document.
  - Fold 4 trained, and folds 3–4 were scored, at `a4b6221`, which only added the resume fix.
- **Score check:** Dice recomputed from the saved float16 maps matches the fold scoring to within
  9.1e-5 per case.
- **Saved maps:** I0 (E0-style) and I2 (TTA) probability maps for all 484 subjects are in
  `outputs/cv_e1_4ch_2026-11-17/oof_probs/` (16 GB, not in git).

## 6. Caveats

- **New baseline.** E1 + D1 is now the baseline that later step-E variants are compared against.
  Its mean is **0.9092** and its bottom decile **0.7621**. Later variants start from the
  4-channel recipe.
- **D1's inference was not re-chosen for E1's models.** The pre-registration fixed it in advance.
  Whether a different configuration would suit E1 better is untested. The saved I0 and I2 maps
  would allow a cross-fitted check, but the other D1 variants would need new inference.
- **One training seed.** The bootstrap CI covers the sampling of subjects, not training-run
  randomness, so part of a +0.0056 difference could be seed noise. All five folds improving
  (+0.0027 to +0.0076) makes a pure seed effect unlikely, but it was not measured.
- **74 cases got worse by more than 0.005**, mostly through more missed tumor. The net gain is
  real under the ship rule, but E1 is not uniformly better.
- **Not comparable** with BraTS/MSD leaderboard numbers (whole tumor only, different data splits).
