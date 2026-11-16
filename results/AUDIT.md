# U-Net audit — 2026-11-15

Scope: section 6 of `docs/handoff-unet-investigation-2026-11-14.md`. Branch `audit/unet`, not pushed. **No training was run.** Nothing under `outputs/` was overwritten or deleted. Every regenerated result is in `outputs/audit_2026-11-15/`, and copies of the small per-case CSVs are in `results/audit_2026-11-15/`. The April 2026 numbers below are quoted as history. They have not been edited anywhere.

## Answers in brief

| Question | Answer |
|---|---|
| Does the ensemble reproduce 0.894 / 0.925? | **Yes, exactly.** Regenerated: mean 0.8939, median 0.9245. All 96 per-case Dice values equal the April CSV (max difference < 1e-4). |
| Do V1 and V2 reproduce? | **Not with the April ensemble settings.** V1 gives 0.8915 vs 0.8928 in April, and V2 gives 0.8905 vs 0.8916. See §7. **With flip TTA on, both reproduce April exactly** (0 of 96 cases differ). So the April single-model numbers used TTA while the April ensemble did not, and the April V1 / V2 / ensemble table did not compare like with like. |
| Is the split clean? | **Yes.** 388 + 96 = 484, no subject in both lists. The split files are byte-identical in the 2.5D reference run, V1, V2 and every 2D/2.5D run. Neither 3D run used the random-split fallback. |
| Is it a held-out test set? | **No.** The 96 chose the checkpoints, the threshold, the post-processing, the V1/V2/ensemble choice, the 2D channel ablation, and the poster panels (§5). Call them "validation". |
| Which case is the poster's "worst" panel? | **BRATS_178 under V1, evaluated without largest-CC (Dice 0.7436).** It is the 6th-lowest case in that evaluation. The actual worst there was BRATS_381 at 0.555. Under V2, whose numbers the poster reports, the worst is BRATS_381 at 0.237 (§8). |

## 1. What "0.894" measures

- Mean volumetric Dice of **whole tumor** (`label > 0`).
- **96 validation subjects** of MSD Task01 (the frozen `val_subjects.txt`).
- Model: the V1 and V2 best checkpoints, with their sigmoid probability maps averaged.
- Inference: 128³ sliding window, 50 % overlap, uniform (count-map) blending, AMP, no TTA.
- Post-processing: threshold 0.5, then keep only the largest connected component.
- SD convention: the April tables used `np.std` with ddof=0 (0.080 / 0.095). The poster's "SD 0.096" for V2 is the same data with ddof=1. Both are shown in §6.

The 96 subjects also made every selection decision listed in §5. So the number is a **validation score with mild optimism**, not a test score. §5 sizes each source of optimism where the files allow it; every measured effect is about 0.001–0.002, well inside the ±0.015 bootstrap CI.

## 2. Environment and how to reproduce

| | |
|---|---|
| GPU | RTX 3090, CUDA 12.1, cuDNN 9.1 |
| Python / torch | 3.12.9, torch 2.5.1+cu121, numpy 2.4.6, scipy 1.18.1, pandas 3.0.3 |
| Interpreter note | The default `python` on this PC (3.14) has a **CPU-only** torch. The CUDA torch is under Python 3.12, which has no scipy. The audit ran in a scratch venv (`py -3.12 -m venv --system-site-packages`, plus `pip install scipy`). The global Python was not touched. |
| Code | Commit `8d44f85` (the `--tta` runs used `36d1194`). Each run's `run_config.json` records the commit and the checkpoint sha256. Five of the six configs are tagged `-dirty`. The only uncommitted file under `src/` at the time was the new, separate `audit_3d.py`, which `evaluate_3d.py` does not import. It is committed together with this document. |

```
python src/evaluation/evaluate_3d.py --model v1       --out-dir outputs/audit_2026-11-15/v1
python src/evaluation/evaluate_3d.py --model v2       --out-dir outputs/audit_2026-11-15/v2
python src/evaluation/evaluate_3d.py --model ensemble --out-dir outputs/audit_2026-11-15/ensemble
python src/evaluation/evaluate_3d.py --model v2 --checkpoint outputs/3d_unet_v2/checkpoints/latest_model_3d_unet_v2.pt --out-dir outputs/audit_2026-11-15/v2_latest
python src/evaluation/evaluate_3d.py --model v1 --tta --out-dir outputs/audit_2026-11-15/v1_tta
python src/evaluation/evaluate_3d.py --model v2 --tta --out-dir outputs/audit_2026-11-15/v2_tta
python src/evaluation/audit_3d.py      # reads files only; writes results/audit_2026-11-15/
```

Each non-TTA run takes 4–6 minutes. `evaluate_3d.py` now refuses to write into a directory that already holds results, so the April CSVs cannot be overwritten by accident.

## 3. Code missing from git (fixed)

`.gitignore` had `data/`, which also matched `src/data/`. So `dataset_3d.py`, `dataset_3d_v2.py`, `preprocess_3d.py` and seven other modules imported by the training scripts were never committed. The patterns are now anchored (`/data/`, `/outputs/`), and the 10 `src/data/*.py` files are committed in `6a9633c`. No MRI data was added; `data/` and `outputs/` at the repo root are still ignored.

## 4. Split integrity

| Source | train | val | overlap | union | = all 484 processed | train sha256 | val sha256 |
|---|---|---|---|---|---|---|---|
| 2.5D reference run | 388 | 96 | 0 | 484 | yes | 2fc023aefd2a | dfd9743500fa |
| 3d_unet_v1 | 388 | 96 | 0 | 484 | yes | 2fc023aefd2a | dfd9743500fa |
| 3d_unet_v2 | 388 | 96 | 0 | 484 | yes | 2fc023aefd2a | dfd9743500fa |

- `data/raw/.../labelsTr` and `data/processed_3d` hold the same 484 subject IDs.
- The three split copies are identical, including order. Both 3D runs loaded the frozen files, so the fallback-seed concern (42 vs 41) did not occur, and V1 and V2 trained on the same 388.
- The same `val_subjects.txt` (md5 `9991fbba…`) is in every 2D/2.5D run directory: `flair_only`, `flair_t1gd_binary`, `flair_t1gd_25d_binary`, `flair_t1gd_binary_smalllesion`, `flair_t1gd_binary_tversky`, and the runtime runs.
- The poster's "387 training / 96" is a typo for 388 / 96.
- Not checked: whether any MSD Task01 patients appear twice under different IDs. That needs image-level deduplication and is out of scope here.

## 5. What the 96 validation subjects were used for

| Use | Which subjects | Evidence | Measured effect on the reported number |
|---|---|---|---|
| **Choosing the checkpoint** (best epoch by sliding-window Dice, threshold 0.5, no CC) | **The first 5 in file order: BRATS_370, BRATS_157, BRATS_042, BRATS_046, BRATS_229.** This is 5 for both V1 and V2, not 10. | Recomputing mean no-CC Dice on val[:5] reproduces the logged `sw_val_dice` to ≤1e-6 for V1 epoch 45, V2 epoch 43 and V2 epoch 50. val[:10] misses by 0.016–0.023. | **V1: none.** Its best checkpoint is byte-identical to the last epoch's (epoch 45 = final logged epoch). **V2:** epoch 43 vs final epoch 50 is +0.0009 mean Dice (paired 95 % CI −0.0009 to +0.0030). |
| Patch-level val loss and Dice each epoch | All 96 (random patches) | `training_log.csv` | Logged only. The code selects on SW Dice when it is available, which it always was. |
| Threshold 0.5 | All 96 | `threshold_sweep_3d.csv` per model; `ENSEMBLE_ANALYSIS.md` "Optimal threshold" | Ensemble: 0.5 beats 0.4 by 0.0010 and 0.6 by 0.0018. |
| Largest-CC on; TTA on for the single models, off for the ensemble ("Phase 1 enhancements") | All 96 | `evaluate_3d.py` comments; exact reproduction in §7 | Largest-CC **lowers** the mean on these 96 (see §9), so this choice did not inflate the number. TTA adds about +0.001 to the single models. |
| V1 vs V2 vs ensemble as the headline | All 96 | `ENSEMBLE_ANALYSIS.md` "Use the ensemble for all poster figures" | Ensemble minus V1 = +0.0024 (CI −0.0012 to +0.0049); ensemble minus V2 = +0.0034 (CI −0.0011 to +0.0100). Neither difference is distinguishable from noise. |
| 2D/2.5D design choices, including the FLAIR + T1gd channel ablation | All 96 | Identical `val_subjects.txt` in every 2D/2.5D run directory (§4) | Not measurable from the files. The 3D input channels were inherited from a choice made on these 96. |
| Poster and README example panels | All 96 | §8 | Illustration only, but the label "worst" is wrong. |

**Conclusion:** no subject leaked between training and validation. The leak is selection: every decision was made on the same 96 used for reporting. Each selection effect that could be measured is small (≤0.002) compared with the sampling uncertainty (bootstrap CI half-width ≈0.016). A number that can honestly be called "held-out" still needs the protocol in handoff step C (cross-validation with selection inside each fold, or an external set).

## 6. Regenerated results — "audit 2026-11-15" (thr 0.5, n = 96, validation subjects)

Headline configuration (largest-CC on, no TTA). SD is ddof=1; ddof=0 is in brackets for comparison with the April tables. The bootstrap uses 10,000 resamples of subjects with seed 20261115.

| Model | Mean | 95 % CI | Median | IQR | SD | Min | ≥0.80 | ≥0.90 |
|---|---|---|---|---|---|---|---|---|
| V1 (best = epoch 45) | 0.8915 | 0.8757 – 0.9062 | 0.9222 | 0.0704 | 0.0767 [0.0763] | 0.6005 (BRATS_381) | 82.3 % | 62.5 % |
| V2 (best = epoch 43) | 0.8905 | 0.8692 – 0.9080 | 0.9223 | 0.0720 | 0.0971 [0.0965] | 0.2253 (BRATS_381) | 85.4 % | 66.7 % |
| **Ensemble V1 + V2** | **0.8939** | **0.8769 – 0.9092** | **0.9245** | 0.0734 | 0.0802 [0.0798] | 0.4728 (BRATS_381) | 83.3 % | 66.7 % |
| V2 epoch 50 (sensitivity only) | 0.8897 | 0.8675 – 0.9071 | 0.9213 | 0.0741 | 0.1018 [0.1013] | 0.1575 (BRATS_381) | 85.4 % | 66.7 % |

For comparison only, the same runs **without** largest-CC (this table chooses nothing):

| Model | Mean | 95 % CI | Median | Min |
|---|---|---|---|---|
| V1 | 0.8942 | 0.8787 – 0.9092 | 0.9230 | 0.5551 |
| V2 | 0.8948 | 0.8744 – 0.9109 | 0.9232 | 0.2255 |
| Ensemble | 0.8976 | 0.8811 – 0.9122 | 0.9259 | 0.4705 |

Per-case tables: `results/audit_2026-11-15/{v1,v2,ensemble,v2_latest}_subject_metrics.csv`. Columns: `dice_3d` (headline), `dice_nocc`, Dice at each threshold, predicted voxels with and without CC, and the number of predicted components. Summary: `results/audit_2026-11-15/summary.csv`.

**History (April 2026, unchanged):** V1 0.894 / 0.923 / SD 0.076 and V2 0.892 / 0.922 / 0.095, both *with flip TTA* (§7); ensemble 0.894 / 0.925 / 0.080, *without* TTA (`ENSEMBLE_ANALYSIS.md`, Apr 3 2026). SBMT poster: V2 + TTA, 0.892 / 0.922 / SD 0.096.

## 7. April vs regenerated: reproduction

| Model | April mean | Regenerated mean | April median | Regenerated median | Max per-case diff | Cases differing > 1e-3 |
|---|---|---|---|---|---|---|
| V1 | 0.8928 | 0.8915 | 0.9225 | 0.9222 | 0.0299 | 45 / 96 |
| V2 | 0.8916 | 0.8905 | 0.9223 | 0.9223 | 0.0114 | 40 / 96 |
| **Ensemble** | **0.8939** | **0.8939** | **0.9245** | **0.9245** | **0.0000** | **0 / 96** |

- The ensemble matches case for case. That shows the same two checkpoint files (mtimes Apr 2 16:00 and Apr 3 04:15, unchanged since) and the same inference code still give April's numbers. It also rules out library or GPU nondeterminism as the cause of the single-model gaps.
- The April single-model CSVs (V1 written Apr 2 21:43, V2 written Apr 3 09:40) match no no-TTA variant: not at any threshold from 0.3 to 0.7, and not with CC on or off.
- The ensemble CSV (Apr 3 10:31) was written after TTA was switched off ("TTA doubles inference time; ensemble already covers variance"), so the likely explanation is that the single-model CSVs were produced with flip TTA on.

**Tested and confirmed.** V1 and V2 re-run with flip TTA (original + 3 single-axis flips, averaged; otherwise identical):

| Model | April mean | Regenerated with TTA | April median | Regenerated with TTA | Max per-case diff | Cases differing > 1e-3 | Max threshold-sweep diff |
|---|---|---|---|---|---|---|---|
| V1 + TTA | 0.8928 | 0.8928 | 0.9225 | 0.9225 | 0.0000 | 0 / 96 | 0.0000 |
| V2 + TTA | 0.8916 | 0.8916 | 0.9223 | 0.9223 | 0.0000 | 0 / 96 | 0.0000 |

Consequences:
- **Every April number reproduces exactly once its configuration is known.** V1 and V2 used TTA + largest-CC. The ensemble used no TTA + largest-CC.
- **The April V1 / V2 / ensemble table compared different inference setups.** The ensemble's apparent edge over V1 (0.894 vs 0.893 mean, 0.925 vs 0.923 median) was set against TTA-boosted single models. Like for like without TTA (§6), the ensemble is +0.0024 over V1 and +0.0034 over V2. Neither gap is distinguishable from noise.
- **The poster's V2 figures** (0.892 / 0.922 / SD 0.096, 85 % ≥ 0.80, 69 % ≥ 0.90) are **V2 + TTA + CC**. The poster methods text does not mention TTA.
- **TTA itself adds +0.0013 (V1) and +0.0011 (V2)** in mean Dice on these 96 subjects. These figures come from the same 96, so they are observations, not a basis for choosing TTA.

Per-case files: `results/audit_2026-11-15/{v1_tta,v2_tta}_subject_metrics.csv`.

## 8. The poster's "best" and "worst" panels

- The two overlay images in `CICI Poster.pptx` (Apr 18 2026, the source of `CICI Poster.pdf`, Apr 22) are **byte-identical** (md5) to:
  - `outputs/3d_unet_v1/visualizations/overlays/best/BRATS_051.png`, titled "3D Dice = 0.9644"
  - `outputs/3d_unet_v1/visualizations/overlays/worst/BRATS_178.png`, titled "3D Dice = 0.7436"
- Those printed values equal the regenerated **V1 Dice without largest-CC** exactly: BRATS_178 0.7436, BRATS_051 0.9644. The V1 `best/` and `worst/` folders match the top and bottom 10 by no-CC Dice (10/10 each). The README's "worst subject BRATS_381 = 0.555" is the same evaluation (regenerated no-CC: 0.5551).
- So the poster panels come from an earlier **V1, no-CC** evaluation (overlays written Apr 2 17:39). The headline text beside them is **V2, with CC**.
- **BRATS_178 was not the worst case** in that evaluation. It was 6th-lowest, behind BRATS_381 (0.555), BRATS_096 (0.684), BRATS_078 (0.732), BRATS_101 (0.736) and BRATS_270 (0.743). In the April V2 CSV, whose numbers the poster states, it is 6th-lowest at 0.778, and the worst is BRATS_381 at 0.237.
- The handoff guessed BRATS_078 under V2 (0.742). That guess was wrong; the hash match settles it.
- Accurate caption for the existing panel: "a below-median example (BRATS_178, V1, Dice 0.74)". It should not be called "worst".

## 9. Observations for the error analysis (step B)

These are observations only. They come from the 96 subjects, so per the protocol they **must not** be used to choose a configuration. Any change has to be decided on training folds.

- **Largest-CC hurts multifocal tumors and lowers the mean.** CC minus no-CC is −0.0027 for V1, −0.0043 for V2 and −0.0037 for the ensemble (CIs cross zero). CC cuts Dice by more than 0.01 on 5–7 cases and raises it by more than 0.01 on 3–5.
- **BRATS_292 is mainly a post-processing casualty, not a model failure.** No-CC Dice is 0.911 (V1), 0.909 (V2) and 0.912 (ensemble). Keeping only the largest component deletes about 54k predicted voxels, and the score falls to about 0.70 in every model. BRATS_154 (0.937 → 0.789) and BRATS_318 (0.88 → 0.79) behave the same way.
- **BRATS_381 is a real miss.** It is low with or without CC: ensemble 0.470 no-CC, 0.473 CC. V2 epoch 50 scores 0.158 on it, so it is also the case most sensitive to the checkpoint. Still open: model miss or label problem. Use surface Dice and a visual review, as in handoff step B.
- **BRATS_096** (0.67, one component, unaffected by CC) and **BRATS_078** (0.74, 12–14 components) are the other persistent low cases.
- **V1 vs V2 is a wash.** The paired mean difference is +0.0010 (CI −0.0056 to +0.0108). V1 is better on 37 subjects and V2 on 59.
- **Mean Dice is flat across thresholds 0.4–0.6** (range ≤0.002 for every model), so the threshold choice hardly matters.

## 10. Claims in docs and on the poster that the code or files contradict

| Where | Claim | What the code or files show |
|---|---|---|
| README (config table, glossary, training summary) | Per-epoch SW validation on **10** subjects | **5** (val[:5]) for both V1 and V2, reproduced to 1e-6 (§5) |
| README config table | V1 batch size 2 | Code and `ENSEMBLE_ANALYSIS.md` say 4. Not verifiable from the files. |
| README / ENSEMBLE_ANALYSIS | V1 50 epochs / 45 epochs | The log has 45 epochs; the README says it crashed at 43. Best = last = epoch 45. The cosine schedule was set for 50 epochs, so V1 never reached the end of its schedule. |
| ENSEMBLE_ANALYSIS, poster | "held-out" | Validation subjects that also made the selection decisions (§5) |
| ENSEMBLE_ANALYSIS | BRATS_381 "a data problem", 0.473 "the ceiling" | Not supported; still open (§9) |
| ENSEMBLE_ANALYSIS | BRATS_292 "edge case … no amount of architectural improvement resolves" | Mostly the largest-CC step: 0.91 without it (§9) |
| ENSEMBLE_ANALYSIS | The ensemble is "bounded above by the better individual model at each subject" | False for averaged probabilities followed by thresholding and CC. The ensemble beats **both** members on 35 / 96 cases (e.g. BRATS_101: ensemble 0.8015, V1 0.8005, V2 0.6898). |
| Poster | 387 training subjects | 388 |
| Poster methods | "Gaussian-weighted overlap" | Uniform count-map averaging (`sliding_window.py`) |
| Poster methods | AdamW | Adam (`train_3d*.py`) |
| Poster methods | "90° rotations … elastic deformation" | Random flips only (V1), plus Gaussian noise and brightness (V2) (`src/data/dataset_3d*.py`) |
| Poster | Worst panel (0.74) | 6th-lowest, from a different model (V1) and post-processing (no CC) than the headline (§8) |
| Poster | "~0.928 SOTA", "within 2–3 points of nnU-Net" | Different test set and protocol. Not comparable (handoff §2). |

None of these documents were edited. They are history, and changing public numbers or captions is the owner's call.

## 11. Not done in this audit

- No training, TTA tuning, threshold tuning or post-processing change.
- No surface Dice / HD95, and no visual or label review of the failure cases (handoff step B).
- No image-level duplicate check across MSD subjects.
- README, ENSEMBLE_ANALYSIS.md, the poster and the site were not changed.

## Commits on `audit/unet`

- `6a9633c` anchor `.gitignore`; commit `src/data/*.py`
- `8d44f85` `evaluate_3d.py`: `--model`/`--out-dir`/`--checkpoint`, overwrite guard, richer per-case CSV
- `36d1194` `evaluate_3d.py`: `--tta` flag (default off)
- (this commit) `audit_3d.py`, `results/AUDIT.md`, `results/RUNS.md`, `results/audit_2026-11-15/`
