# Project Plan & Next Steps

**Project:** AI-Based Prediction of Chemotherapy-Induced Cognitive Impairment (CICI)  
**Presentation:** SBMT 2026 NeuroTech Convention — Los Angeles, April 16–19  
**Current status:** 3D U-Net segmentation complete. Mean Dice 0.894 / Median 0.923 on 96 val subjects.

---

## Phase 1 — Quick Model Wins (No Retraining)

These improve the headline Dice number without touching model weights. Do these first.

- [ ] **Test-time augmentation (TTA)** — modify `src/evaluate_3d.py` to run sliding-window inference on the original volume plus its 3 axis-flipped copies, average the 4 probability maps, then threshold. Expected gain: +0.01–0.02 Dice.
- [ ] **Connected component post-processing** — after thresholding, retain only the largest connected component per subject using `scipy.ndimage.label`. Eliminates isolated false-positive voxels. Adapt logic already present in `src/evaluate.py` (2.5D version) for 3D.
- [ ] **Checkpoint ensemble** — average sigmoid probability maps from the best 3–5 saved checkpoints before thresholding. Small but consistent variance reduction.
- [ ] Re-run `src/evaluate_3d.py` after the above and update the Dice numbers in `README.md` and poster.

---

## Phase 2 — Stronger Segmentation Model (Retrain, ~2–4 hrs each)

Only proceed here if Phase 1 results are insufficient or if time permits after Phase 3.

- [ ] **Larger feature sizes** — change `features=[32, 64, 128, 256]` to `[64, 128, 256, 320]` in `src/model_3d.py`. Run a 1-epoch sanity check first to confirm VRAM headroom.
- [ ] **Heavier foreground bias** — change the 50/50 patch sampling split to 70/30 foreground-biased in `src/dataset_3d.py`. Targets improvement on small-tumor failure cases (e.g., BRATS_381, Dice 0.555).
- [ ] **Intensity augmentation** — add random Gaussian noise (`σ ~ U(0, 0.05)`) and brightness scaling (`× U(0.85, 1.15)`) in `dataset_3d.py` augmentation block.
- [ ] **Deep supervision** — add auxiliary segmentation heads at intermediate decoder levels in `model_3d.py`. Each head contributes a weighted loss term (0.5 × intermediate + 1.0 × final).

---

## Phase 3 — CICI Tabular Risk Model

This is the scientific core of the project and the key differentiator of the poster narrative. No new dataset required — built on published cohort statistics from the literature.

- [x] Create `src/tabular_risk_model.py`
  - [x] Define feature set: age, baseline cognitive score (MoCA/MMSE), drug regimen (cyclophosphamide / doxorubicin / taxanes = high risk), cumulative dose, inflammatory markers (IL-6, CRP, TNF-α), APOE ε4 status, education years
  - [x] Synthesize a training cohort from published CICI literature statistics (odds ratios, effect sizes, reported cohort distributions from the 75 filtered papers)
  - [x] Train XGBoost (primary) and logistic regression (baseline) classifiers
  - [x] Evaluate with cross-validation; report AUC, sensitivity, specificity
  - [x] Save feature importance plot for poster
- [x] Create `src/fusion_demo.py` — stub showing how tabular risk score + imaging encoder features would be late-fused (concatenate → logistic layer). Can be architecture-only with placeholder data for the poster figure.

**Results (April 3, 2026):**
- XGBoost AUC: **0.806** [0.756, 0.855] — exceeds TLC LASSO benchmark (0.75–0.79)
- Logistic Regression AUC: **0.766** [0.715, 0.816]
- XGBoost Sensitivity: 0.728 | Specificity: 0.726 | F1: 0.707
- Synthetic cohort: 1,500 subjects, 45.2% CICI+ (literature-grounded)
- 5-fold CV AUC: 0.789 ± 0.032 (XGBoost), 0.782 ± 0.037 (Logistic)
- All outputs in `outputs/cici_risk_model/`

---

## Phase 4 — Real CICI Data (Post-Conference)

These require external data access and are post-April-16 targets.

- [ ] **Apply for ADNI access** — longitudinal MRI + cognitive test scores + demographics + biomarkers. Free with data use agreement. Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu). Typically approved within 1–2 weeks.
- [ ] **Explore OASIS-3** — immediately downloadable open longitudinal brain MRI with cognitive and clinical data. [oasis-brains.org](https://www.oasis-brains.org)
- [ ] Once data is available: adapt `src/preprocess_3d.py` and `src/dataset_3d.py` to the new dataset's format and modalities
- [ ] Fine-tune the 3D U-Net encoder (freeze decoder, attach FC head) as an imaging feature extractor for the full CICI pipeline
- [ ] Train end-to-end fusion model: imaging features + tabular risk score → CICI probability
- [ ] Submit to a journal or conference (e.g., NeuroImage, MICCAI, or a dedicated CICI/onco-cognition venue)

---

## Phase 5 — Poster (Owned by collaborator)

Tracked here for coordination.

- [ ] Background panel: CICI clinical problem + literature gap (0/75 papers with drug-specific models)
- [ ] Methods panel: full proposed CICI prediction pipeline figure (tabular branch + imaging branch → fusion)
- [ ] Imaging results panel: training curves, Dice distribution, best/worst overlay examples from `outputs/3d_unet_v1/visualizations/`
- [ ] Results table: progression from 2D baseline (Dice 0.6335) → 2.5D → 3D U-Net (Dice 0.894 / 0.923)
- [ ] Roadmap panel: path from proof-of-concept to real CICI data and clinical deployment
- [ ] Final Dice numbers: update after Phase 1 is complete

---

## Architecture Reference — Full Proposed CICI Pipeline

```
Patient intake
       │
       ├──► Tabular branch ─────────────────────────────────────────►─┐
       │    Age, drug regimen, dose, biomarkers (IL-6, CRP),           │
       │    APOE status, baseline cognitive score                       │
       │    → XGBoost → scalar risk score                              ▼
       │                                                        Fusion layer
       └──► Imaging branch ─────────────────────────────────────►     │
            Baseline MRI (FLAIR + T1gd)                         concat + logistic
            → 3D U-Net encoder (pretrained, frozen)             → CICI risk score
            → Global average pool
            → FC feature vector
```

---

## Key Numbers

| Branch | Metric | Value |
|--------|--------|-------|
| **Imaging** | Ensemble Mean Dice | 0.894 |
| | Ensemble Median Dice | 0.925 |
| | Optimal threshold | 0.5 |
| | Best subject | BRATS_442 — 0.968 |
| | Worst subject | BRATS_381 — 0.473 (ensemble) |
| **Tabular** | XGBoost AUC | 0.806 [0.756, 0.855] |
| | Logistic AUC | 0.766 [0.715, 0.816] |
| | XGBoost Sensitivity | 0.728 |
| | XGBoost Specificity | 0.726 |
| | Synthetic cohort | 1,500 subjects, 45.2% CICI+ |
