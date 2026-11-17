# HANDOFF — brain-lesion-detection U-Net investigation — 2026-11-17 — #2

Follows `docs/handoff-unet-investigation-2026-11-14.md` (#1, the original brief). Read #1 §3–§5 for
background; everything still binding from it is carried below.

## 1. Where we are

**Repo:** `D:\Brain Lesion Detection` on the owner's PC.

- Remote `origin` = github.com/mattin-tasbihgoo/SBMT-Mattin-Aarush (public; the old name
  `brain-lesion-detection` redirects there).
- `main` = `c387a2c`, pushed: audit, step B, step C protocol, E0, D1.
- Current branch is **`e1/four-channels`** at `defc881`, **local only, not pushed**. The working
  tree is clean apart from this file.
- `data/` and `outputs/` are gitignored (`/data/`, `/outputs/`); they exist only on this PC.

**Environment:**

- The default `python` (3.14) has CPU-only torch. Use the Python 3.12 venv:
  `C:\Users\Matti\AppData\Local\Temp\claude\d710941cd935982c\452a7ee7-4279-4837-a75b-610056f7fce7\scratchpad\venv312\Scripts\python.exe`
  (torch 2.5.1+cu121 from system Python 3.12, plus scipy, nibabel, matplotlib).
- If that temp folder is gone, recreate the venv:
  `py -3.12 -m venv --system-site-packages <dir>`, then `<dir>\Scripts\pip install scipy nibabel matplotlib`.
- Hardware:
  - RTX 3090 24 GB; i7-14700KF (28 threads); 128 GB RAM.
  - The page file is fixed at 1 GB, so the Windows commit limit is about 129 GB. Other apps hold
    about 60 GB.
  - `D:` (repo + data) is a SATA **hard disk**; `C:` is an NVMe drive with only about 84 GB free.

**Results so far.** All numbers are whole tumor (`label > 0`). Out-of-fold (OOF) means every
subject is scored by the one fold model that never trained on it.

| Name (keep the dates) | Set | Mean Dice | 95% CI | Median | Write-up |
|---|---|---|---|---|---|
| April 2026 V1 / V2 / ensemble (history) | 96 validation subjects | 0.8928 / 0.8916 / 0.8939 | — | 0.9225 / 0.9223 / 0.9245 | `results/AUDIT.md` |
| E0, 5-fold CV, V1 recipe, 2026-11-16 | OOF, n = 484 | 0.8972 | 0.8908–0.9035 | 0.9205 | `results/CV_E0.md` |
| **E0 + D1 inference, 5-fold CV, 2026-11-16 (current baseline)** | OOF, n = 484 | **0.9036** | 0.8979–0.9091 | 0.9218 | `results/D1.md` |
| E1 (4 channels), in progress | OOF, n = 484 | — | — | — | — |

- **Current baseline (E0 + D1):** E0 fold models, flip TTA (original + 3 single-axis flips),
  count-map averaging, 50% overlap, threshold 0.5, keep every connected component that is at
  least 10% of the largest.
  - Bottom-decile mean (the 48 lowest cases): 0.7542.
  - Per-case table: `results/d1_2026-11-16/per_case_crossfit.csv`.
- **Persistent low cases (E0 + D1):** BRATS_381 0.546, BRATS_096 0.698, BRATS_270 0.704,
  BRATS_078 0.734, BRATS_178 0.773.
- **Cases D1 made worse** (a false second region is now kept): BRATS_477 0.921 → 0.802; BRATS_132,
  BRATS_428 and BRATS_168 each lose about 0.05.

**The E1 run (started 2026-11-17 10:51, expected to finish around 20:00–21:00):**

- **What it is:** the E0 recipe with 4 input channels (FLAIR, T1gd, T1w, T2w), scored with the
  frozen D1 inference configuration. It was pre-registered in `results/RUNS.md` before training.
- **Where it runs:** a visible PowerShell window titled "E1 4-channel 5-fold CV"; closing that
  window stops the run.
- **Log:** `outputs/cv_e1_4ch_2026-11-17/run.log`. The run ends with the line `ALL DONE`, or with
  `FAILED` / `giving up`.
- **Output:** on success it writes `results/cv_e1_2026-11-17/{per_case_oof.csv, summary.csv,
  ship_rule.csv}` and prints the ship-rule result in the log.
- **Resuming:** if it was interrupted (reboot), rerun the command below. Finished folds are skipped
  and an unfinished fold resumes from its last finished epoch.

```
cd "D:\Brain Lesion Detection"
<venv python> src/training/run_cv.py --run-dir outputs/cv_e1_4ch_2026-11-17 --tag cv_e1_2026-11-17 --recipe E1 --workers 14
```

**Live files:**

| File | What it does |
|---|---|
| `results/PROTOCOL.md` | the evaluation protocol: folds, set roles, decision rules, metrics, ship rule |
| `results/protocol/folds_5fold_seed20261115.csv` | frozen folds (sha256 `eaf1cf46…6165714`); never edit |
| `results/RUNS.md` | lab notebook: one row per run, plus the D1 and E1 pre-registrations |
| `src/training/train_cv.py` | trains one fold. `--recipe E0` (default) or `E1`. Final-epoch checkpoint, resumable, `--bench N` for timing |
| `src/training/run_cv.py` | runs all 5 folds, then the report. Retries a crashed fold with fewer workers; keeps Windows awake |
| `src/evaluation/score_fold_e.py` | E-variant fold scoring with the D1 inference. Saves I0 and I2 maps to `<run>/oof_probs/` |
| `src/evaluation/compare_cv.py` | E-variant OOF aggregate plus the ship rule against E0 + D1 |
| `src/evaluation/evaluate_cv.py`, `src/evaluation/cv_report.py` | E0-only fold scoring and report |
| `src/evaluation/infer_variants_cv.py`, `src/evaluation/d1_crossfit.py` | D1 inference variants and cross-fitting |
| `src/evaluation/error_analysis_3d.py` | `case_metrics()`: Dice, surface Dice @1/2 mm, HD95, ASSD, subregion misses |
| `src/data/preprocess_3d_4ch.py` | builds `data/processed_3d_4ch/images` (float16, 33 GB; `--check` verifies channels 0–1 equal E0's inputs) |

The large intermediates live in `outputs/` and are not in git:

- `outputs/cv_e0_v1recipe_2026-11-15/`: E0 fold checkpoints and float16 OOF probability maps (8 GB).
- `outputs/d1_inference_2026-11-16/probs/I1..I5/`: the D1 inference variants.
- `outputs/error_analysis_2026-11-15/case_figures/`: step B review panels.

**Status:**

- **Done and pushed:**
  - audit (`results/AUDIT.md`);
  - step B error analysis (`results/ERROR_ANALYSIS.md`);
  - step C protocol;
  - E0 baseline;
  - D1, which shipped (+0.0064, CI +0.0037 to +0.0095).
- **Running:** E1.
- **Broken:** nothing known.

## 2. Corrections log

- **Never run `python` bare; use the 3.12 venv.** The default 3.14 install has CPU-only torch
  and silently runs on the CPU.
- **Use at most 14 DataLoader workers, and run nothing else on the GPU during training.**
  - On Windows each spawned worker re-imports torch/CUDA and commits about 4 GB.
  - 20 workers plus a concurrent evaluation hit the commit limit:
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 68.1 MiB`.
  - One training run needs about 15 GB of VRAM, so two runs never fit.
  - 14 workers are already GPU-bound: 0.41 s per iteration equals the pure GPU step.
- **Don't memory-map the training images or switch to channels_last to "speed up".** Both were
  benchmarked. mmap was 2× slower on this hard disk, and channels_last gave no gain.
- **Write files from Python with `encoding="utf-8"`.** The default here is cp1252. A `→` in
  `Path.write_text` raised mid-write and truncated `results/RUNS.md` to 0 bytes (restored from
  git).
- **Check the logs before believing a run is "done".** A user report of finished runs once came
  while one run was at 54/96.
- **April V1 and V2 used flip TTA; the April ensemble did not.** With `evaluate_3d.py --tta` they
  reproduce exactly; without it they don't (gap of about 0.001).
- **April checkpoints were chosen on the first 5 validation subjects** (BRATS_370, 157, 042, 046,
  229), not 10.
- **Don't call the 96 April subjects "held-out" or a test set.** They chose the checkpoint,
  threshold, post-processing, TTA and model. Call them "validation" (the owner is relabelling old
  materials).
- **Don't explain BRATS_381 by its size.** It is a model miss on faint, diffuse edema. It stays the
  lowest case out-of-fold too (0.546).
- **The poster's "worst" panel is BRATS_178, from V1 without largest-CC (0.7436).** It is not the
  worst case. The poster is not being reused; no caption fixes are needed.
- **Don't run `src/data/preprocess_3d.py` as is.** Its `PROJECT_ROOT` is `parents[1]` (= `src/`),
  so its paths are wrong. The existing `data/processed_3d` is correct and verified.
- **Use the stored E0 maps as the E0/D1 baseline; don't recompute them.**
  - Re-running inference gives up to 1.7e-3 probability differences (GPU nondeterminism), about
    1e-5 in Dice.
  - Dice from the float16 maps differs from float32 by at most 7e-4 per case.
- **Don't compare against nnU-Net or "~0.928"** unless it is nnU-Net trained and scored on these
  same folds. Published numbers use a different test set and often per-region scoring.

## 3. Constraints and settled decisions

- **Never choose anything on the subjects being scored.** That covers checkpoints, thresholds,
  post-processing and design changes. This is the owner's rule and the reason the investigation
  exists.
- **Every run follows `results/PROTOCOL.md`:**
  - 5-fold CV over all 484 subjects, with the frozen folds file;
  - final-epoch checkpoint; no test-fold metrics during training;
  - inference options chosen by cross-fitting (for each fold, chosen on the other four folds only).
- **Ship rule against the current baseline** (now E0 + D1):
  - the paired-bootstrap 95% CI of the mean Dice difference (10,000 resamples, seed 20261115) is
    entirely above 0, and
  - the bottom-decile mean is not lower (now ≥ 0.7542).
  - Rationale: a change must beat noise and must not hurt the worst cases.
- **Pre-register the candidate list in `RUNS.md` and commit it before scoring.** Disclose anything
  already seen that shaped the list.
- **Step-E variants use the frozen D1 inference configuration** unless D1 is re-run by
  cross-fitting.
- **No silent exclusions.** Every subject stays in the headline. Suspected label problems
  (BRATS_078, 096, 075) get documented, and at most a second, clearly labelled number.
- **April numbers are dated history.** New results get new, dated names and folders
  (`outputs/<name>_<date>/`, `results/<name>_<date>/`); never overwrite old ones.
- **Report the whole distribution:** mean, median, IQR, SD, min, the fraction ≥ 0.80 and ≥ 0.90,
  the 10k bootstrap CI, per-fold means and a per-case CSV.
- **Log every run in `results/RUNS.md`,** kept or not.
- **Commit locally; push only with the owner's explicit OK each time.** New numbers become public
  on push. Ask before deleting or overwriting checkpoints or outputs, and before posting any new
  number.
- **GPU use is pre-authorized.** Still say up front which runs need an overnight slot. One 5-fold
  training run takes about 10–11 h.
- **Scope:** the brain-tumor segmentation proof of concept from the chemo-brain (CICI) project.
  Self-directed, not USC Khan Lab work, not a CICI predictor. Don't change that framing in any doc.
- **The 5-model fold ensemble gets no score on these 484** (each model trained on 80% of them).
  It is for demos and the website viewer only.
- **Store 4-channel data as float16 on `D:`.** The AMP first convolution casts to float16 with the
  same rounding, so this is lossless. Float32 would be 70 GB, which won't stay in the RAM cache and
  is too large for `C:`.

## 4. Rejected, and why not to revisit

- **Dropping hard cases from the reported set** — it hides failures; label concerns get
  documented, not deleted.
- **Picking checkpoints, thresholds or post-processing on the reporting set, or on pooled OOF
  results** — that is the leak this investigation exists to remove. Use cross-fitting.
- **Chasing 0.90 without error analysis** — blind tuning finds noise.
- **Reporting an nnU-Net run as the owner's model** — it is a reference ceiling, credited to
  nnU-Net.
- **Mixing old and new numbers in one table without dates** — the April figures are history.
- **Largest-connected-component post-processing** — it deleted correctly found second lesions
  (for example BRATS_258 0.635 vs 0.895). D1 replaced it with "keep components ≥ 10% of the
  largest".
- **Gaussian-weighted sliding window** — slightly worse than count averaging for every threshold
  and rule in D1.
- **Thresholds other than 0.5** — 0.4 and 0.6 are each about 0.001 worse; 0.3 is worse still.
- **Training two folds concurrently** — about 15 GB of VRAM each; they don't fit.
- **Early stopping or a "best" checkpoint** — it needs a validation signal from scored subjects.
  The final epoch of the fixed cosine schedule is the rule.

## 5. Open questions

- **E1's outcome** (ship or not). The descriptive check is whether the edema miss share drops from
  13%.
- **Next step-E variant after E1**, each one an overnight run judged against the current baseline.
  Options already listed: spatial augmentation (rotation, scaling, elastic, gamma), longer
  training, region targets (whole tumor, tumor core, enhancing tumor). No order has been chosen.
- **Step F:** nnU-Net 3d_fullres on the same folds as a reference ceiling (about 1–2 days of GPU).
  Not started.
- **Which number goes on the owner's site and essays.** The owner decides; the candidate is the
  latest shipped CV number, quoted with its name and date.
- **Possible duplicate patients in MSD Task01.** No patient IDs are given, so this is unchecked.
  If found, folds would need patient-level grouping.
- **Label review of BRATS_078, 096 and 075** needs an expert; it can't be judged from code.
- **Whether to publish the step B MRI review panels.** They would need CC-BY-SA 4.0 attribution
  (Antonelli et al., Nature Communications 2022, plus the BraTS papers).
- **Whether to make an external test set** (a later BraTS release, checked for patient overlap
  with Task01).
- **Machine changes (owner's call):** enlarge the page file, or move `data/` to the NVMe.

## 6. Next action

When the E1 run finishes (`ALL DONE` in `outputs/cv_e1_4ch_2026-11-17/run.log`), write it up the
way `results/D1.md` was written:

- Create `results/CV_E1.md` with the headline table vs E0 + D1, the ship-rule result, subregion
  miss shares, the biggest per-case gains and losses, and the persistent failures.
- Copy the fold logs and configs into `results/cv_e1_2026-11-17/fold_logs/`.
- Add the E1 row to `results/RUNS.md`.
- Commit on `e1/four-channels`, then ask the owner before merging or pushing.

If the log shows `FAILED` or `giving up` instead, read the traceback in `run.log` and resume with
the command in §1.

**Done when:** `results/CV_E1.md` and the `RUNS.md` row are committed locally on
`e1/four-channels`, and the owner has been told the ship/no-ship result and asked whether to push.
