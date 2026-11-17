# Lab notebook — runs

One row per run, kept or not. "Set" names the subjects scored. **val96** is the frozen 96-subject validation list (`val_subjects.txt`, md5 `9991fbba…`). It also made the selection decisions (`results/AUDIT.md` §5), so val96 scores are validation scores, not test scores. Dice is whole tumor at threshold 0.5. "CC" means only the largest connected component is kept. SD uses ddof=1. Bootstrap CIs are in `results/audit_2026-11-15/summary.csv`.

## History (April 2026; not re-run as training, numbers as recorded then)

| Date | Run | Config | Set | Mean | Median | Note |
|---|---|---|---|---|---|---|
| 2026-04-02 | 3d_unet_v1 training | [32,64,128,256], 50/50 fg, flips, batch 4, 45 of 50 epochs, best epoch chosen on val96[:5] (best = last = 45) | train 388 | — | — | `outputs/3d_unet_v1` |
| 2026-04-03 | 3d_unet_v2 training | [64,128,256,320] + DS, 70/30 fg, flips + noise/brightness, batch 3, 50 epochs, seed 41, best epoch chosen on val96[:5] (epoch 43) | train 388 | — | — | `outputs/3d_unet_v2` |
| 2026-04-02 | V1 eval | flip TTA + CC | val96 | 0.893 | 0.923 | TTA confirmed by exact re-run 2026-11-15 |
| 2026-04-03 | V2 eval | flip TTA + CC | val96 | 0.892 | 0.922 | SBMT poster figures; TTA confirmed 2026-11-15 |
| 2026-04-03 | Ensemble eval | V1 + V2 mean prob, no TTA, CC | val96 | 0.894 | 0.925 | Site/essay figures |

## Audit 2026-11-15 (inference only, no training; RTX 3090, torch 2.5.1+cu121)

| Date | Commit | Run (`outputs/audit_2026-11-15/…`) | Config | Set | Mean | Median | SD | Min | ≥0.80 | ≥0.90 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | 8d44f85 | `v1` | V1 best, no TTA, CC | val96 | 0.8915 | 0.9222 | 0.0767 | 0.6005 | 82.3 % | 62.5 % | Reference, like for like with the ensemble |
| 2026-11-15 | 8d44f85 | `v2` | V2 best (ep 43), no TTA, CC | val96 | 0.8905 | 0.9223 | 0.0971 | 0.2253 | 85.4 % | 66.7 % | Reference, like for like with the ensemble |
| 2026-11-15 | 8d44f85 | `ensemble` | V1 + V2 best, no TTA, CC | val96 | 0.8939 | 0.9245 | 0.0802 | 0.4728 | 83.3 % | 66.7 % | **Reproduces April ensemble exactly** (0 / 96 cases differ) |
| 2026-11-15 | 8d44f85 | `v2_latest` | V2 epoch 50, no TTA, CC | val96 | 0.8897 | 0.9213 | 0.1018 | 0.1575 | 85.4 % | 66.7 % | Sensitivity only: checkpoint-selection effect +0.0009 |
| 2026-11-15 | 36d1194 | `v1_tta` | V1 best, flip TTA, CC | val96 | 0.8928 | 0.9225 | 0.0747 | 0.6304 | 83.3 % | 62.5 % | **Reproduces April V1 exactly**: April V1 used TTA |
| 2026-11-15 | 36d1194 | `v2_tta` | V2 best, flip TTA, CC | val96 | 0.8916 | 0.9223 | 0.0959 | 0.2367 | 85.4 % | 68.8 % | **Reproduces April V2 / poster exactly**: April V2 used TTA |

## Error analysis 2026-11-15 (step B; inference only, no training)

| Date | Commit | Run | Config | Set | Result | Verdict |
|---|---|---|---|---|---|---|
| 2026-11-15 | analysis/error-b | `outputs/error_analysis_2026-11-15/probs/{v1,v2}` | V1 + V2 best, no TTA; probabilities saved (float32) | val96 | Dice reproduces the audit exactly (all 3 models × CC / no CC × 96) | Analysis only. Ensemble + CC: sDice@2mm median 0.909, HD95 median 3.5 mm. See `results/ERROR_ANALYSIS.md` |

## 5-fold CV (step C protocol, `results/PROTOCOL.md`)

| Date | Commit | Run | Config | Folds | Result | Verdict |
|---|---|---|---|---|---|---|
| 2026-11-15 22:44 → 2026-11-16 08:58 (a 22:27 attempt crashed in epoch 1, Windows commit limit) | 4f08374 | E0 `outputs/cv_e0_v1recipe_2026-11-15` | April V1 recipe, 50 epochs, final-epoch checkpoint; inference SW 0.5 overlap, no TTA, thr 0.5, largest CC | 5-fold, all 484 OOF | mean **0.8972** (CI 0.8908–0.9035), median 0.9205, SD 0.0717, min 0.521, ≥0.80 89.5 %, ≥0.90 66.1 %; bottom decile 0.7242; April-96 subset 0.8935 | **Baseline** for D/E. See `results/CV_E0.md`. No-CC stored (0.9015) but not a frozen config; goes to D1 cross-fitting |

### D1 pre-registration (written 2026-11-16, before any D1 scoring)

Inference-only options on the 5 E0 fold models; each subject is predicted only by the model of its own test fold. Choice by **cross-fitting** (`PROTOCOL.md` §5.1): for fold k, the configuration with the highest mean Dice on the other four folds' OOF subjects is chosen and applied to fold k. Selection criterion: mean whole-tumor Dice. Ties: first in the order listed.

- **Inference** (6): I0 count-map averaging, 50% overlap, no TTA (= E0); I1 Gaussian weighting (σ = patch/8, nnU-Net style), 50% overlap; I2 flip TTA (original + 3 single-axis flips) with count averaging; I3 flip TTA + Gaussian; I4 count averaging, 75% overlap; I5 Gaussian, 75% overlap.
- **Threshold** (4): 0.3, 0.4, 0.5, 0.6.
- **Post-processing** (7): none; largest connected component; drop components < 100, < 1,000, < 5,000 voxels; keep components ≥ 10 %, ≥ 25 % of the largest (6-connectivity throughout).
- 6 × 4 × 7 = 168 configurations. Fold-ensembling is excluded (cannot be scored honestly on these 484).
- Ship rule vs E0 (I0, 0.5, largest CC): paired bootstrap 95 % CI of the mean Dice difference > 0 (10,000 resamples, seed 20261115) and bottom-decile mean not lower than 0.7242.
- Disclosure: E0's pooled OOF table (no-CC 0.9015 vs CC 0.8972) was seen before this list was written, which is why component-size rules are on it. Cross-fitting keeps each fold's choice blind to that fold.

| Date | Commit | Run | Config | Folds | Result | Verdict |
|---|---|---|---|---|---|---|
| 2026-11-16 | fc0ad91 | D1 `outputs/d1_inference_2026-11-16` | 168 pre-registered inference × threshold × post-processing configs on the E0 fold models; cross-fitted choice | 5-fold, all 484 OOF | every fold chose I2 (flip TTA, count avg) + thr 0.5 + keep components ≥ 10 % of largest; mean **0.9036** (CI 0.8979–0.9091), median 0.9218, SD 0.0628, min 0.546, ≥0.80 92.1 %, ≥0.90 68.4 %; paired diff vs E0 +0.0064 (CI +0.0037 to +0.0095); bottom decile 0.7542 | **Ships.** New baseline = E0 + D1 inference. See `results/D1.md` |
