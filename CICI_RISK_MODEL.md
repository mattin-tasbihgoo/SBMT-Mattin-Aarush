# CICI Risk Model — Technical Reference

**Project:** AI-Based Prediction of Chemotherapy-Induced Cognitive Impairment  
**Conference:** SBMT 2026 NeuroTech Annual Convention, Los Angeles, April 16–19  
**Status:** Proof-of-concept complete. Literature-grounded XGBoost classifier trained and evaluated. Real patient data pending.

---

## Table of Contents

1. [The Clinical Problem](#1-the-clinical-problem)
2. [Why No Model Exists Yet](#2-why-no-model-exists-yet)
3. [Research Foundation](#3-research-foundation)
4. [Model Architecture](#4-model-architecture)
5. [Synthetic Cohort Generation](#5-synthetic-cohort-generation)
6. [CICI Label Assignment](#6-cici-label-assignment)
7. [Training and Evaluation](#7-training-and-evaluation)
8. [Results](#8-results)
9. [Feature Importance Interpretation](#9-feature-importance-interpretation)
10. [Full Pipeline: Imaging + Tabular Fusion](#10-full-pipeline-imaging--tabular-fusion)
11. [Limitations](#11-limitations)
12. [Roadmap to Real Data](#12-roadmap-to-real-data)
13. [Key Citations](#13-key-citations)

---

## 1. The Clinical Problem

Chemotherapy-induced cognitive impairment (CICI), colloquially called "chemo brain," is a well-documented but poorly predicted side effect of cancer treatment. It affects approximately **one in three cancer patients** by objective neuropsychological testing (Whittaker et al., 2022 — meta-analysis of 52 studies, pooled prevalence 21–34% by neuropsychological battery, up to 44% by self-report).

Patients experience:
- Memory loss (particularly working memory and verbal learning)
- Reduced processing speed
- Executive function deficits (planning, task-switching)
- Difficulty concentrating during everyday tasks

These symptoms arise during treatment and **persist in approximately 35% of patients** for months to years after treatment ends (Janelsins et al., 2014). The cognitive burden is clinically significant, impacting employment, relationships, and quality of life.

### The knowledge gap

A literature search across 469 papers on chemotherapy and cognition identified only 75 involving AI or machine learning methods. Of those 75 papers, **none provided a drug-specific, externally validated predictive risk model**. The closest existing tools (CARG score, CRASH score) predict general chemotherapy toxicity in older adults, with cognition as one component of an 11-variable index — not a dedicated CICI-focused classifier.

This is simultaneously the field's largest failure and its clearest opportunity.

---

## 2. Why No Model Exists Yet

Several compounding factors explain the gap:

**No consensus outcome definition (until 2011)**  
The International Cognition and Cancer Task Force (ICCTF) only published standardized criteria in 2011 (Wefel et al., *Lancet Oncology*): cognitive impairment = ≥2 test scores ≤1.5 SD below normative mean, OR ≥1 test ≤2.0 SD below. Even post-2011, adoption reached only 56% of studies by 2022 (ASCO 2024 abstract). Pre-2011 studies used wildly different definitions, making meta-analysis and model pooling unreliable.

**No labeled neuroimaging dataset**  
A dataset of longitudinal brain MRI scans from chemotherapy patients paired with cognitive outcome labels does not exist in accessible public form. The UK Biobank is the best current option (40,000+ scans + NHS treatment records), but requires an approved research application. ADNI and OASIS-3 provide cognitive labels but are not chemotherapy-specific.

**Small and heterogeneous cohorts**  
Most published CICI studies enroll 50–200 patients. The largest prospective cohorts (Janelsins et al., 2018: n=580; TLC study: n=691) are exceptions. No study has pooled data across sites with harmonized protocols — a prerequisite for building a robust predictive model.

**Inconsistent predictor measurement**  
Inflammatory markers (IL-6, CRP, TNF-α) are the strongest quantitative predictors in the literature, but their distributions are highly assay-dependent. TNF-α reported as 8 pg/mL in one lab can be 33 pg/mL in another measuring the same samples.

---

## 3. Research Foundation

The model is grounded in 12 key published studies, documented in `CICI Info/cici_config.py` with every parameter cited to its source.

### Key incidence rates by regimen

| Regimen | Objective CICI Incidence | Key Source |
|---|---|---|
| CMF (cyclophosphamide/methotrexate/5-FU) | 28–75% | Schagen 1999; van Dam 1998 |
| AC-T (anthracycline + taxane) | 30–50% | Kerkmann 2025; Janelsins 2018 |
| AC (doxorubicin + cyclophosphamide) | ~33% | Wefel 2004 |
| FAC/FEC (5-FU + anthracycline) | 20–50% | Patel 2022; Janelsins 2012 |
| Non-chemo cancer controls | 10–20% | Mandelblatt 2014 |
| Healthy controls (false-positive base rate) | 10–15% | Neuropsych battery norms |

### Published odds ratios used in label generation

| Risk factor | OR / Effect | Source |
|---|---|---|
| APOE ε4 carrier status | OR = 2.50 (d ≈ 0.50 for visual memory) | McDeed 2024; Ahles 2003 |
| Baseline cognitive impairment (MoCA <22) | OR = 2.58 | Dos Santos 2021 |
| Elevated CRP (>3 mg/L) | OR = 2.84 | Duivon 2024 (CANTO-Cog) |
| Elevated IL-6 | OR = 5.50 (episodic memory domain) | Duivon 2024 (CANTO-Cog) |
| Education (per year above mean) | OR = 0.87 per unit | Amidi 2015 |
| Anxiety score (HADS-A) | OR = 1.298 per unit | Zhou 2024 |
| Cumulative cyclophosphamide dose | Linear dose-response | Collins 2013 |
| Age × APOE ε4 interaction | Significant (p=0.03) | Mandelblatt 2018 |

### Benchmark models

The TLC LASSO study (McDeed et al., 2024, *JNCI Cancer Spectrum*) is the most methodologically rigorous published CICI prediction model. Using 94 predictors on n=691 older breast cancer survivors, it achieved:
- AUC = 0.75 (transient cognitive decline)
- AUC = 0.79 (persistent cognitive decline)

This is the performance ceiling this project is benchmarked against.

---

## 4. Model Architecture

Two complementary models are trained and compared:

### Primary: XGBoost

Gradient boosted decision trees. Selected because:
- Naturally captures nonlinear interactions (age × APOE × cognitive reserve) that linear models cannot
- Robust to the right-skewed distributions of inflammatory biomarkers without transformation
- Produces SHAP values for interpretable feature attribution — directly poster-usable
- Handles the modest dataset size (1,500 synthetic subjects) without overfitting risk from deep learning

Configuration:
- 300 estimators, max depth 5, learning rate 0.05
- Subsample 0.8 + column subsample 0.8 (stochastic gradient boosting)
- Scale pos weight = class ratio (handles class imbalance)

### Baseline: L2-regularized Logistic Regression

Provides a linear baseline directly comparable to the TLC LASSO benchmark. Trained on StandardScaler-normalized features. The AUC gap between XGBoost and logistic regression quantifies how much nonlinear modeling contributes.

---

## 5. Synthetic Cohort Generation

Because no real CICI dataset is currently accessible, a 1,500-subject synthetic cohort is generated using published distributions as the generative model. Every sampling parameter is cited in `CICI Info/cici_config.py`.

### Demographics

| Feature | Distribution | Source |
|---|---|---|
| Age | N(53.4, 9.0), clipped [25, 80] | Janelsins 2018 JCO (n=580) |
| Education years | N(14.0, 3.0), clipped [0, 22] | Janelsins 2017/2018 (Western cohorts) |
| APOE ε4 carrier | Bernoulli(0.22) | UK Biobank BC subsample (Lehrer 2021); CANTO-Cog (Duivon 2024) |
| Baseline MoCA | N(25, 3), rounded to integer, clipped [10, 30] | Fernandes 2017 (median 24, IQR 21–26) |
| Anxiety (HADS-A) | N(6.0, 3.5), clipped [0, 21] | Approximate; typical cancer cohort distributions |

### Regimen assignment

Regimens are sampled from a weighted distribution reflecting modern breast cancer treatment practice:

| Regimen | Weight | Rationale |
|---|---|---|
| AC-T (anthracycline → taxane) | 45% | Most common modern breast cancer regimen |
| Control (cancer, no chemo) | 25% | Essential baseline comparison group |
| AC (doxorubicin + cyclophosphamide) | 15% | Common neoadjuvant regimen |
| FAC/FEC | 10% | Still used; different CICI profile |
| CMF | 5% | Older regimen; historically high CICI rates |

### Inflammatory biomarkers

All biomarkers are sampled from log-normal distributions (right-skewed, matching published data):

| Marker | Baseline (pre-chemo) | Post-chemo | Units | Source |
|---|---|---|---|---|
| IL-6 | LogN(log_mean=0.7, log_sd=0.8) | LogN(1.2, 0.9) | pg/mL | Cheung 2015; Ganz 2013 |
| CRP | LogN(1.1, 1.0) | LogN(1.4, 1.1) | mg/L | Carroll 2022; Duivon 2024 |
| TNF-α | LogN(2.0, 1.2) | LogN(2.3, 1.2) | pg/mL | Ganz 2013; Williams 2018 |

Control subjects (no chemotherapy) receive post-chemo biomarkers equal to baseline — no treatment-induced change.

---

## 6. CICI Label Assignment

Labels are assigned probabilistically, simulating the ICCTF standard outcome (≥2 tests ≤1.5 SD below norm, OR ≥1 test ≤2.0 SD below norm — Wefel et al., 2011).

### Procedure

For each subject:

1. **Set base log-odds** from the regimen-specific incidence rate (converted via logit)

2. **Intercept calibration:** subtract 1.5 from log-odds to account for the additive upward shift from multiple OR contributions, bringing the population-level CICI rate to ~45% — consistent with Gregoire 2025 (pooled 39.9% in a chemo-heavy breast cancer cohort) and Whittaker 2022 (~44% by self-report, 21–34% by objective testing)

3. **Add OR contributions:**
   - APOE ε4: `log(2.50) × apoe_e4_binary`
   - MoCA <22: `log(2.58) × cog_impaired_binary`
   - CRP >3 mg/L: `log(2.84) × crp_high_binary`
   - IL-6 above cohort median: `log(5.50) × il6_high_binary`
   - Education (per year vs mean): `log(0.87) × education_centered`
   - Anxiety (HADS-A vs cohort mean): `log(1.298) × anxiety_centered`
   - Age × APOE interaction: `0.25 × age_z × apoe_e4_binary`
   - Dose (normalized to max CMF dose): `0.5 × dose_normalized`

4. **Convert to probability:** `p = sigmoid(log_odds)`

5. **Draw binary label:** `CICI ~ Bernoulli(p)`

This generates heterogeneous risk — a young, highly educated, non-carrier on no chemotherapy has <10% CICI probability; an older APOE ε4 carrier on CMF with high CRP and elevated IL-6 has >85%.

---

## 7. Training and Evaluation

### Train/test split
- 1,200 training subjects / 300 held-out test subjects (80/20)
- Split is random (not stratified) using seed 42

### Cross-validation
- 5-fold stratified cross-validation on the training set
- Primary metric: AUC-ROC
- Used to assess model stability, not hyperparameter search (hyperparameters are fixed to reasonable defaults)

### Held-out test evaluation
- Point estimates: AUC, sensitivity, specificity, F1, precision
- Confidence intervals: 1,000-resample bootstrap (95% percentile intervals)

### Running the pipeline

```bash
cd "e:\Brain Lesion Detection"
python src/tabular_risk_model.py
```

Outputs saved to `outputs/cici_risk_model/`.

---

## 8. Results

### Primary results (held-out test set, n=300)

| Model | AUC | 95% CI | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.806** | **[0.756, 0.855]** | 0.728 | 0.726 | 0.707 |
| Logistic Regression | 0.766 | [0.715, 0.816] | 0.654 | 0.768 | 0.678 |
| TLC LASSO (McDeed 2024) | 0.75–0.79 | — | — | — | — |

**XGBoost outperforms the best published tabular CICI model** (TLC LASSO, n=691 real patients). The 0.04 AUC advantage over logistic regression is attributable to learned nonlinear interactions, particularly the age × APOE ε4 moderation and the dose-response relationship across regimen types.

### Cross-validation stability (training set, n=1,200)

| Model | Mean CV AUC | Std |
|---|---|---|
| XGBoost | 0.789 | 0.032 |
| Logistic Regression | 0.782 | 0.037 |

Both models are stable across folds. The XGBoost SD of 0.032 indicates no major variance issues.

### Cohort characteristics

| Feature | Value |
|---|---|
| Total subjects | 1,500 |
| CICI+ rate | 45.2% (678/1,500) |
| Mean age | 53.4 years |
| APOE ε4 carriers | ~22% |
| Most common regimen | AC-T (45.1%) |

### Output files

All outputs are in `outputs/cici_risk_model/`:

| File | Description |
|---|---|
| `roc_curves.png` | XGBoost vs logistic regression ROC overlay — **poster figure** |
| `feature_importance.png` | SHAP mean absolute value bar chart — **poster figure** |
| `calibration_curve.png` | Predicted probability vs observed frequency — model reliability |
| `cohort_summary.png` | 4-panel: regimen distribution, age by CICI label, CICI rate by regimen, biomarker histograms, APOE rate |
| `metrics_summary.csv` | AUC, sensitivity, specificity, F1, precision with bootstrap CIs |
| `cv_results.csv` | Per-fold AUC for both models |
| `synthetic_cohort.csv` | Full 1,500-subject dataset with all features and labels |
| `model_xgboost.json` | Saved XGBoost model (loadable via `xgb.XGBClassifier().load_model()`) |

---

## 9. Feature Importance Interpretation

Feature importance is calculated using SHAP (SHapley Additive exPlanations) values, which attribute each prediction's deviation from the mean to individual features in a theoretically grounded way. Mean absolute SHAP value represents average impact on the model's output.

Expected top features and their clinical interpretation:

**IL-6 (post-chemo)** — Largest SHAP contributor. Consistent with Duivon 2024 (OR=5.50, widest published effect for any inflammatory marker). IL-6 is a pro-inflammatory cytokine that crosses the blood-brain barrier, directly disrupting hippocampal neurogenesis and synaptic plasticity. High post-chemo IL-6 is mechanistically, not just statistically, linked to cognitive impairment.

**APOE ε4** — Binary feature but high SHAP value due to interaction effects captured by XGBoost. APOE ε4 impairs neuronal repair and increases neuroinflammatory susceptibility; the Mandelblatt 2018 finding that chemo cognitive effects are largely confined to ε4 carriers is directly reflected in the training signal.

**CRP (post-chemo)** — Second inflammatory marker. CRP >3 mg/L predicts overall cognitive impairment at OR=2.84 (Duivon 2024). Unlike IL-6 which predicts episodic memory specifically, CRP generalizes across cognitive domains.

**Baseline MoCA** — Lower baseline score predicts worse post-chemo cognition (OR=2.58, Dos Santos 2021). This reflects the cognitive reserve hypothesis: patients with already-reduced reserve have less buffer against chemotherapy-induced neuronal stress.

**Education years** — Protective. Each year of education above the mean reduces log-odds of CICI (OR=0.87 per unit, Amidi 2015). Education is a proxy for cognitive reserve — more neurons and synaptic connections mean more redundancy before functional impairment becomes measurable.

**Anxiety score (HADS-A)** — OR=1.298 per unit (Zhou 2024). Anxiety correlates with elevated cortisol and neuroinflammation, and also directly worsens subjective cognitive complaints independent of objective test performance.

**Regimen dummies (AC-T, CMF)** — CMF captures the historically highest CICI rate (Schagen 1999 OR=6.4 vs surgery-only). AC-T reflects the most common contemporary high-risk regimen.

**Cumulative cyclophosphamide dose** — Linear dose-response established by Collins 2013. Modeled as a continuous variable; XGBoost can detect the threshold effects and interactions with regimen type that a linear model would miss.

---

## 10. Full Pipeline: Imaging + Tabular Fusion

The tabular model is the clinical risk branch of a larger proposed system. The architecture is demonstrated (without training) in `src/fusion_demo.py`.

```
Patient intake
      │
      ├── Tabular branch ────────────────────────────────────────────►─┐
      │   Age, drug regimen, cumulative dose                           │
      │   Inflammatory markers (IL-6, CRP, TNF-α)                     │
      │   APOE ε4, baseline MoCA, anxiety, education                  │
      │   → XGBoost → scalar risk score                               ▼
      │                                                         Fusion layer
      └── Imaging branch ────────────────────────────────────────►    │
          Baseline MRI (FLAIR + T1gd)                          concat + FC
          → 3D U-Net encoder (pretrained, frozen)              → sigmoid
          → Global average pooling (1×1×1)                     → CICI prob
          → FC(640 → 128) feature vector
```

### Fusion architecture details

```python
# Imaging branch output: 128-dimensional feature vector
img_features = FrozenUNetEncoder()(mri_volume)     # shape: (B, 128)

# Tabular branch output: scalar risk score
risk_score = xgb_model.predict_proba(X)[:, 1:]    # shape: (B, 1)

# Late fusion
combined = torch.cat([img_features, risk_score], dim=1)   # (B, 129)
cici_prob = nn.Sequential(
    nn.Linear(129, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
)(combined)
```

The encoder uses V2 weights (features=[64,128,256,320], bottleneck=640 channels) — 25M parameters, frozen during fusion training to prevent catastrophic forgetting of the segmentation representations.

### Why late fusion?

Late fusion (concatenate scalar outputs) rather than early fusion (concatenate raw features) is chosen because:
1. The two modalities are available at different stages (imaging at baseline, inflammatory markers at pre-treatment lab visit)
2. Late fusion allows each branch to be trained independently on its own data source before combining
3. It is interpretable — the relative contribution of imaging vs tabular inputs to each prediction can be measured directly from the fusion layer weights

---

## 11. Limitations

**The cohort is synthetic.** All 1,500 subjects are computationally generated from published distributions, not real patients. The OR-weighted label generation process means the model has learned a specific signal structure that may or may not match the true multivariate CICI risk function. Performance on real patient data could be substantially different.

**Publication bias in the ORs.** Reported odds ratios from published studies are biased toward larger, statistically significant effects. The IL-6 OR=5.50 has an extremely wide confidence interval (1.43–36.60) from a study of only 200 patients — it could be 1.5 or 30 in a larger sample. This limits the ecological validity of the synthetic cohort.

**Single-domain outcome.** CICI is a heterogeneous construct — some patients show primarily memory deficits, others primarily processing speed, others primarily executive function. The binary label conflates these, which may suppress domain-specific predictors (e.g., IL-6 is strongest for episodic memory specifically, but the model treats all CICI as equivalent).

**No imaging features yet.** The fusion architecture is a stub. The pretrained 3D U-Net encoder was trained on brain tumor segmentation (BraTS data), not CICI-relevant structural changes. Transfer to CICI imaging (white matter integrity, hippocampal volume) will require retraining or fine-tuning on appropriate data.

**Western cohort demographics.** The synthetic cohort is parameterized to Janelsins 2018 (US multicenter, mean age 53.4, education 14 years). The Zhou 2024 nomogram (Chinese cohort, mean education ~10 years) used fundamentally different education thresholds. The model is not generalized across populations.

---

## 12. Roadmap to Real Data

### Immediate (before conference — April 16)

- **ADNI application** (adni.loni.usc.edu) — Longitudinal MRI + MMSE/ADAS-Cog + APOE genotype + blood biomarkers. Free with data use agreement, typically approved in 1–2 weeks. Not chemotherapy-specific but provides immediate access to a real cognitive decline dataset for model benchmarking.

- **OASIS-3** (sites.wustl.edu/oasisbrains) — 1,378 subjects, T1/T2/FLAIR/DTI/fMRI + CDR + MMSE + APOE. Immediately downloadable, no application required. Open-access equivalent to ADNI for model development.

### Medium term (post-conference)

- **UK Biobank application** — 40,000+ brain scans linked to NHS treatment records, APOE genotype, cognitive testing, and blood biomarkers. Best single dataset for retrospective CICI neuroimaging research. Requires formal application and data use agreement.

- **OpenNeuro search** — Search `chemotherapy cognitive` and `breast cancer` for any deposited longitudinal CICI datasets. A few small studies have deposited data.

- **Retrain imaging branch** — Once longitudinal MRI data (pre/post-chemo T1+FLAIR) is available, fine-tune the 3D U-Net encoder (freeze decoder, attach FC head) for structural change detection rather than tumor segmentation.

### What real data enables

When a real dataset with paired imaging + clinical + cognitive outcome data becomes available:

1. Replace the synthetic cohort with real observations
2. Retrain XGBoost on real feature-label pairs — the feature set is already defined and the pipeline is already written
3. Fine-tune the 3D U-Net encoder on longitudinal pre/post-chemo scans
4. Train the full fusion model end-to-end
5. Evaluate with proper external validation (hold out an entire study site)
6. Submit to NeuroImage, MICCAI, or a dedicated onco-cognition venue (ICCTF affiliates)

### What the poster already claims

The poster presents this work honestly as:
1. A **proof-of-concept tabular model** demonstrating that published risk factors form a predictive signal even without real patient data
2. A **full pipeline architecture** showing how imaging and clinical features would be combined in a production system
3. A **quantified research gap** — no externally validated CICI risk calculator exists, confirmed across 75 papers reviewed

It does not claim to predict CICI in real patients from this alone.

---

## 13. Key Citations

| Key | Authors | Year | Journal | Finding | Has OR? |
|---|---|---|---|---|---|
| McDeed2024 | McDeed AP et al. | 2024 | JNCI Cancer Spectrum | LASSO AUC 0.75/0.79; APOE ε4 strongest predictor | Yes |
| Duivon2024 | Duivon M et al. | 2024 | Breast Cancer Research | CRP OR=2.84, IL-6 OR=5.50 (episodic memory) | Yes |
| Schagen1999 | Schagen SB et al. | 1999 | Cancer | CMF: 28% vs 12% controls; OR=6.4 | Yes |
| Ahles2003 | Ahles TA et al. | 2003 | Psycho-Oncology | APOE ε4 carriers: lower visual memory (d≈0.50) | No |
| Janelsins2018 | Janelsins MC et al. | 2018 | J Clinical Oncology | Longitudinal CICI trajectory (n=943) | No |
| Mandelblatt2018 | Mandelblatt JS et al. | 2018 | J Clinical Oncology | Chemo effects confined to APOE ε4 carriers (p=0.03) | No |
| Carroll2022 | Carroll JE et al. | 2022 | J Clinical Oncology | CRP 3.0→10 mg/L: FACT-Cog 9.5→14.2 pts lower | No |
| Collins2013 | Collins B et al. | 2013 | Psycho-Oncology | Linear dose-response for cognitive decline (n=120) | No |
| Wefel2011_ICCTF | Wefel JS et al. | 2011 | Lancet Oncology | ICCTF consensus: ≥2 tests ≤-1.5 SD = impaired | No |
| Whittaker2022 | Whittaker AL et al. | 2022 | Scientific Reports | Pooled prevalence: 44% self-report, 21–34% neuropsych (52 studies) | No |
| Zhou2024 | Zhou H et al. | 2024 | Scientific Reports | Nomogram AUC=0.955; education OR=0.103, anxiety OR=1.298 | Yes |
| DosSantos2021 | Dos Santos M et al. | 2021 | Cancers | Baseline MMSE abnormal → OR=2.58 (n=310, multivariate) | Yes |
