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
