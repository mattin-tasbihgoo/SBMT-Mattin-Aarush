# CICI Survival Analysis — Pre-Registration Design Document

**Project:** AI-Based Prediction of Chemotherapy-Induced Cognitive Impairment (CICI)
**Refactor:** Binary classification (XGBoost, AUC 0.806) → Survival analysis (Cox PH, Cox-LASSO, Survival XGBoost)
**Pre-registered:** April 19, 2026
**Status:** LOCKED before any data generation or model fitting. Changes require documented justification in [Section 10](#10-change-control).

---

## 0. Why pre-registration

Every modeling decision below is fixed *before* the synthetic cohort is generated. This is the standard defense against implicit p-hacking, garden-of-forking-paths bias, and post-hoc rationalization — which the Duan et al. 2025 systematic review of CRCI prediction models explicitly called out as endemic in the field. Because the entire data-generating process is under our control, any retrospective tuning of outcome definitions, covariate codings, or model choices would silently amplify apparent performance. Locking the design removes that freedom by construction.

This document derives from `CICI_Survival_Analysis_Evidence_Base.md`. Every numeric parameter traces back to that report.

---

## 1. Scientific objective

**Primary question.** Can a time-to-event model predict *when* a cancer patient will develop CICI using pre-treatment clinical, inflammatory, genetic, and regimen data — and does it outperform (i) a binary landmark classifier at the same horizon, and (ii) the best published CRCI binary model (TLC LASSO, McDeed 2024, AUC 0.75–0.79)?

**Secondary question.** Does a drug-specific survival model (which distinguishes AC, AC-T, CMF, FAC/FEC, platinum, and targeted therapy as separate hazard contributors) outperform a chemo-yes/no binary survival model? This operationalizes the "drug-specific risk prediction" gap identified in the literature review.

**Novelty statement (for manuscript framing).** The Duan et al. 2025 review catalogued nine CRCI prediction models through August 2024. All nine are binary classifiers. None use Cox PH, random survival forest, DeepSurv, or any time-to-event framework. A PubMed search through April 2026 returned zero CICI-specific survival-analysis publications. This refactor is the first.

---

## 2. Outcomes

### Primary outcome — Reliable Change Index (RCI) decline on APE composite

A CICI event is defined as the first assessment visit at which a patient's composite APE (Attention / Processing-speed / Executive function) z-score drops by at least **1.645 SD** from their own individual baseline, after correcting for the expected practice effect. This matches Wefel 2004 and Ahles 2003 and is the most defensible discrete-timestamp operationalization.

Formally: RCI = (X₂ − X₁ − practice_effect) / SE_diff; event = 1 if RCI ≤ −1.645.

### Secondary outcome — FACT-Cog PCI < 54

Subjective cognitive complaints crossing the validated impairment threshold (Van Dyk 2020). Higher event rate and thus higher statistical power, but known to correlate more strongly with depression/anxiety/fatigue than with objective neuropsychological decline. This is a sensitivity outcome, not a co-primary.

### Target event rates (calibration check for synthetic DGP)

| Outcome | Target at 12 mo | Target at 24 mo | Source |
|---|---|---|---|
| RCI on APE | 30% (±5%) | 42% (±5%) | Janelsins 2018 objective decline |
| FACT-Cog PCI < 54 | 45% (±5%) | 55% (±5%) | Whittaker 2022 self-report pooled estimate |

If the simulated cohort does not fall in these ranges after sampling, the baseline hazard scale parameter will be recalibrated (see Section 9).

### Outcomes that are NOT primary

Composite objective-AND-subjective criteria, ICCTF three-test criteria, domain-specific decline (memory-only, executive-only), and any continuous-score endpoint are explicitly *not* primary outcomes in this refactor. They are reserved for post-hoc exploratory analyses, clearly labeled as such.

---

## 3. Follow-up horizon and time scale

- **Administrative censoring horizon (τ):** 24 months from treatment start.
- **Time scale:** time-on-study (months since treatment initiation), not calendar time.
- **Time origin (t = 0):** first chemotherapy infusion (or equivalent index date for controls).
- **Extended horizon sensitivity:** a 60-month analysis is pre-specified as exploratory only, using the Weibull + Gompertz mixture baseline from the evidence base (Section 2 of the report). Any result beyond 24 months is labeled exploratory in the results.

---

## 4. Models

Fit in this order. Results are reported in this order. No model is dropped from the results table based on performance; null or inferior results are reported honestly.

### Primary model
**Cox proportional hazards with APOE ε4 stratification** via lifelines `CoxPHFitter` with `strata=['apoe_e4']`. Stratification is pre-specified because the Mandelblatt 2018 TLC analysis (JCO) reports a significant APOE × chemotherapy × age three-way interaction (p = 0.03), which is strong circumstantial evidence for a time-varying APOE effect and therefore a PH violation. Stratifying the baseline hazard on APOE is the safest pre-registered handling.

### Secondary models (head-to-head with primary)
1. **Cox-LASSO** — lifelines `CoxPHFitter` with L1 penalty, penalty strength chosen by 10-fold CV on the training set. This is the natural regularized extension of McDeed 2024's published benchmark.
2. **Survival XGBoost** — either sksurv `GradientBoostingSurvivalAnalysis` or xgboost with `objective='survival:cox'`. Tuning grid pre-specified: max_depth ∈ {3, 5, 7}, n_estimators ∈ {100, 300, 500}, learning_rate ∈ {0.01, 0.05, 0.1}. Best config selected by 10-fold CV c-index. This is the direct Fork B analog and the headline survival-ML contribution.
3. **Weibull AFT** — lifelines `WeibullAFTFitter`. Fits directly with interval-censored likelihood. Primary fallback when PH fails.

### Binary baseline (for landmark comparison)
**XGBoost binary classifier at the 12-month landmark** — the existing Fork B model (`tabular_risk_model.py`) treated as a fixed-horizon landmark. Trained on the same cohort with the same covariates, event defined as "CICI event observed by 12 months," patients censored before 12 months excluded. This is the apples-to-apples comparison against the binary paradigm.

### Stretch model (only if all above complete by Sprint 3)
**Survival Transformer / DeepSurv** via pycox. Reported only if it adds interpretive value; not required for the main result.

---

## 5. Covariate handling

### Time-varying covariates (mandatory)
- **Cumulative chemotherapy dose.** Must be coded as time-varying (start-stop / counting-process format). Fixed-at-baseline coding produces immortal-time bias and is pre-specified as incorrect. lifelines `CoxTimeVaryingFitter` supports this natively.
- **Endocrine therapy exposure.** Binary indicator that switches 0 → 1 at endocrine-therapy start. Typically begins ~4 months post-chemo.

### Stratification covariates
- **APOE ε4 status.** Stratifies the Cox baseline hazard in the primary model.

### Fixed covariates (baseline only)
Age, education, baseline MoCA, regimen type, baseline inflammatory markers, HER2 status, BMI category, cardiovascular comorbidity, baseline HADS-A anxiety, baseline PHQ-9 depression, baseline FACIT-F fatigue.

### Covariates measured post-baseline but treated as fixed
Post-chemotherapy inflammatory markers (IL-6, CRP, TNF-α) are measured 2–6 weeks after last cycle and coded as fixed-at-baseline for the 0–24 month model. Rationale: acute inflammatory effects are the mechanism of interest in this horizon; by 12–24 months these markers have normalized and their hazard contribution fades. Using the peak post-chemo value captures the exposure window when it matters most. Longer-horizon analyses would require time-varying coding.

### Interactions pre-specified
- **APOE × chemotherapy** (built into the simulation; included in the model as a product term)
- **Age × APOE** (for sensitivity analyses only)

All other interactions are exploratory and clearly labeled as such if reported.

---

## 6. Pre-registered hazard ratios

These are the HR point estimates wired into the synthetic data-generating process. The fitted primary Cox model must recover each within tolerance; otherwise there is a simulation bug. Ported verbatim from the parameter table in `CICI_Survival_Analysis_Evidence_Base.md`.

| Covariate | HR (DGP) | Acceptable recovery range | Role |
|---|---|---|---|
| APOE ε4 carrier | 1.75 | 1.40 – 2.20 | Stratified — not estimated |
| IL-6 elevated (post-chemo, binary) | 2.80 | 2.20 – 3.50 | Fixed predictor |
| CRP > 3 mg/L (binary) | 1.85 | 1.45 – 2.35 | Fixed predictor |
| Baseline MoCA < 22 (binary) | 1.80 | 1.40 – 2.30 | Fixed predictor |
| Education (per year above 14) | 0.92 | 0.88 – 0.96 | Protective, fixed |
| Anxiety (HADS-A, per unit) | 1.20 | 1.12 – 1.30 | Fixed predictor |
| Depression (PHQ-9, per unit) | 1.08 | 1.03 – 1.14 | Fixed predictor |
| Fatigue (per severity unit) | 1.05 | 1.02 – 1.09 | Fixed predictor |
| Age (per 10 years above 50) | 1.35 | 1.20 – 1.52 | Fixed predictor |
| Cumulative dose (per 1000 mg/m² cyclo-eq) | 1.15 | 1.08 – 1.23 | Time-varying |
| CMF vs. no chemo | 3.80 | 2.80 – 5.10 | Regimen dummy |
| AC / AC-T vs. no chemo | 2.40 | 1.85 – 3.10 | Regimen dummy |
| FAC / FEC vs. no chemo | 2.60 | 2.00 – 3.40 | Regimen dummy |
| Platinum vs. no chemo | 2.10 | 1.55 – 2.85 | Regimen dummy |
| Targeted vs. no chemo | 1.40 | 1.00 – 1.95 | Regimen dummy |
| HER2+ | 1.35 | 1.00 – 1.80 | Fixed predictor |
| Cardiovascular comorbidity | 1.40 | 1.05 – 1.85 | Fixed predictor |
| BMI ≥ 30 | 1.30 | 1.00 – 1.70 | Fixed predictor |

Acceptable ranges are approximately ±20% of the DGP HR on the log scale, wider for rarer covariates. Recovery failure triggers a code review, not a parameter change.

---

## 7. Evaluation plan

### Train / test split
- **80 / 20 stratified split** on event status, matching the existing Fork B protocol for direct comparability.
- **Random seed:** 42 (fixed).
- **Bootstrap:** 1,000 resamples on the held-out test set for 95% CIs on all reported metrics.

### Cross-validation
- **10-fold CV on the training set** for all hyperparameter selection. No test-set use during model development.

### Primary metrics
- **Harrell's c-index** (discrimination, 0.5 = random, 1.0 = perfect). Primary headline metric.
- **Uno's c-index** (inverse-probability-of-censoring-weighted alternative). Reported alongside Harrell's.
- **Time-dependent AUC at 6, 12, and 24 months** (via sksurv `cumulative_dynamic_auc`).
- **Calibration plot at 12 and 24 months** (predicted vs. observed survival by risk decile). This is the Duan 2025 credibility differentiator — CRCI models systematically fail to report this.
- **Integrated Brier Score** over 0–24 months (combined calibration and discrimination).

### PH assumption diagnostics
- **Schoenfeld residual test** (lifelines `check_assumptions`) on every non-stratified covariate. Covariates with p < 0.05 are flagged and reported in the supplementary results.

### Success thresholds (set before fitting)

| Metric | Minimum defensible | Target | Rationale |
|---|---|---|---|
| Cox PH c-index | ≥ 0.72 | ≥ 0.77 | Floor = McDeed AUC; target = Musto 2024 Cox PH benchmark |
| Survival XGBoost c-index | ≥ 0.75 | ≥ 0.80 | Target = Musto 2024 XGBoost benchmark |
| Survival XGBoost − Cox PH Δ | ≥ 0.02 | ≥ 0.03 | Must beat the linear model to justify ML |
| Time-dependent AUC at 12 mo | ≥ 0.78 | ≥ 0.82 | Should match or exceed the binary landmark AUC 0.806 |
| Calibration slope | 0.85 – 1.15 | 0.95 – 1.05 | — |
| Integrated Brier Score | ≤ 0.22 | ≤ 0.18 | Standard acceptable threshold |

Results falling below "minimum defensible" will be reported honestly with a candid discussion of probable cause.

---

## 8. Sensitivity analyses (pre-specified)

Four sensitivity analyses are pre-specified and will be reported regardless of whether they support or contradict the primary results.

**S1 — Interval-censoring schedule.** Refit the primary pipeline on three visit schedules:
- Schedule A (dense): 0, 3, 6, 9, 12, 18, 24 months.
- Schedule B (CICI-typical — default for primary): 0, 4, 6, 12, 24 months.
- Schedule C (sparse): 0, 12, 24 months.

Report c-index and HR estimates under each. Expected direction: c-index degrades as interval widens.

**S2 — Competing mortality.** Refit using Fine-Gray subdistribution hazards (sksurv or `cmprsk` via rpy2) in place of cause-specific Cox, under each of three cohort mortality scenarios (non-metastatic BC 4.5% / elderly 7.7% / metastatic 51% at 24 months). Report divergence from the primary cause-specific results.

**S3 — PH-violation handling.** Compare three handlings of APOE ε4: (a) fixed covariate in unstratified Cox, (b) stratified baseline in primary Cox, (c) Weibull AFT. Report hazard-ratio stability and Schoenfeld test outcomes.

**S4 — Outcome robustness.** Refit the entire primary pipeline using FACT-Cog PCI < 54 as the outcome. Report c-index and HR stability versus the RCI-based primary.

---

## 9. Synthetic cohort generation — pre-registered parameters

### Cohort size
**n = 2,500** (up from the 1,500 used in the binary Fork B model). Survival analysis loses effective sample size to censoring; the extra 1,000 patients restore statistical power roughly equivalent to the original 1,500 for a binary outcome.

### Baseline hazard
- **Primary distribution:** Weibull, shape k = 0.85, scale λ = 22 months.
- **Alternative (sensitivity):** log-logistic, α = 10 months, β = 1.3.
- **Extended 0–60 mo (exploratory only):** Weibull + Gompertz mixture.

Calibration check: under the primary hazard and the reference patient profile (no chemo, age 50, APOE−, no covariates elevated), cumulative incidence at 12 months must equal ~30% ± 5% for the RCI outcome.

### Censoring
- Administrative censoring at τ = 24 months.
- Uninformative dropout: 10% per 6-month interval (exponential with rate ≈ 0.0176/month).
- Informative-dropout sensitivity: rate × exp(0.3 × linear_predictor).

### Competing mortality
Not included in the primary cohort. Applied only in sensitivity analysis S2.

### Random seed
`SEED = 42` throughout simulation and analysis.

### Recalibration protocol
If the simulated 12-month event rate falls outside 30% ± 5% for the RCI outcome under the reference profile, the Weibull scale parameter will be recalibrated via bisection (bounded to 18–28 months per the evidence base) until it lands in range. This is the *only* parameter permitted to move post-hoc, and the recalibration is logged.

---

## 10. Change control

After this document is committed to the repo, any modification requires:
1. A new entry in the change log at the bottom of this document.
2. An explicit rationale tied to a reference in `CICI_Survival_Analysis_Evidence_Base.md` or to a specific code-level finding.
3. Re-running any completed analysis whose design depended on the changed parameter.

Allowed *without* amendment:
- Bug fixes in code that do not alter modeling decisions.
- Recalibration of the Weibull scale parameter to hit the 12-month event rate target (Section 9).
- Hyperparameter tuning within the pre-specified grids (Section 4).

Not allowed without amendment:
- Changing primary or secondary outcomes.
- Adding or dropping covariates from the primary model.
- Modifying HR point estimates (Section 6) outside the bug-fix or evidence-update path.
- Changing success thresholds (Section 7).

### Change log

| Date | Section | Change | Rationale |
|---|---|---|---|
| 2026-04-19 | — | Initial pre-registration committed. | Locks design before Sprint 1 code. |

---

## 11. Appendix — Reference materials

- **Evidence base:** `CICI_Survival_Analysis_Evidence_Base.md`
- **Parameter module:** `CICI Info/cici_survival_config.py`
- **Simulation code:** `src/survival/survival_cohort.py`
- **Primary benchmark paper:** Musto H et al. 2024. *ICANN 2024* Springer. [arXiv:2409.16231](https://arxiv.org/abs/2409.16231). Reports Cox PH c-index 0.77, Survival XGBoost c-index 0.80, Survival Transformer c-index 0.85 on ADNI metabolomics → MCI-to-AD progression.
- **Closest CRCI binary benchmark:** McDeed AP et al. 2024. *JNCI Cancer Spectrum* 8(2):pkae019. TLC LASSO, AUC 0.75–0.79.
- **Systematic review establishing the survival gap:** Duan et al. 2025. *Clinical and Experimental Medicine*. [DOI 10.1007/s10238-025-01590-6](https://doi.org/10.1007/s10238-025-01590-6).
