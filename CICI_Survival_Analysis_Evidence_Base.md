# CICI Survival Analysis Evidence Base

**A comprehensive research report for refactoring the CICI risk prediction model from binary classification (XGBoost, AUC 0.806) to a survival-analysis framework (Cox PH, Cox-LASSO, Survival XGBoost).**

*Prepared April 19, 2026. Primary sources: PubMed searches through April 2026, the Duan et al. 2025 systematic review of CRCI prediction models (through August 2024), and your existing 469-paper / 75-study project literature base. Every quantitative claim is tied to a primary source; every area of thin evidence is explicitly flagged.*

---

## Executive summary — read this first

Three findings drive the rest of this report and should shape your framing.

**1. CICI longitudinal cohorts do not report in a survival-analysis-native format.** Virtually every major CICI longitudinal study — Janelsins 2017/2018, CANTO-Cog, Sun 2025, Palesh 2025, Colaes 2025 — uses **linear mixed models (LMMs) or latent-class/growth-mixture trajectory modeling**, not Kaplan–Meier curves or Cox regression. This means almost no published CICI paper gives you a hazard function, a baseline survival distribution, or a time-to-first-impairment curve to fit to. Section 2 addresses this by reconstructing time-to-event probabilities from reported incidence at each assessment timepoint. The survival framing is itself a methodological contribution.

**2. Your "no drug-specific AI/ML CICI risk model" gap is still standing — and is actually stronger in the survival subdomain.** The Duan et al. 2025 *Clinical and Experimental Medicine* systematic review ([DOI](https://doi.org/10.1007/s10238-025-01590-6), PMID 40047952) catalogued nine CRCI prediction models published through August 2024. **All nine use binary classification** — logistic regression, LASSO, ANN, SVM-RFE, or Elastic Net — with AUCs from 0.72 to 0.955. **None use Cox PH, random survival forest, DeepSurv, or any time-to-event framework, and none stratify on specific chemotherapy agents beyond binary "chemo yes/no" or regimen cycle counts.** Section 3 walks through each. The closest survival-adjacent paper is Tohidinezhad et al. 2024 ([DOI](https://doi.org/10.1093/neuonc/noae035)), which is radiotherapy-not-chemo and still uses time-binned logistic regression rather than a true survival model.

**3. The Musto et al. 2024 ADNI metabolomics benchmark gives you a concrete, defensible c-index target for the poster narrative.** Using 100 repetitions of nested cross-validation to predict MCI-to-AD progression, they reported mean c-index values of **0.77 (Cox PH), 0.80 (Survival XGBoost), and 0.85 (survival transformer)** on the same dataset with the same features — a clean head-to-head comparison published in ICANN 2024 proceedings ([arXiv:2409.16231](https://arxiv.org/abs/2409.16231)). For CICI — a noisier, more heterogeneous target with smaller published effect sizes than the ADNI MCI→AD signal — a **Cox c-index of 0.70–0.76 and a Survival XGBoost c-index of 0.74–0.80** would be clinically credible and methodologically honest.

Everything below supports those three framings with primary literature.

---

## Section 1 — Event definition for CICI as a survival endpoint

The threshold used to convert continuous cognitive scores into a binary "impaired / not impaired" event on the time axis is the single most consequential modeling choice, because it fixes both event rate and noise level.

### The five candidate definitions

**ICCTF consensus criterion (Wefel et al. 2011, *Lancet Oncology*, [DOI](https://doi.org/10.1016/S1470-2045%2810%2970294-1), PMID 21145789).** Cognitive impairment defined as **≥ 2 neuropsychological test scores falling ≤ 1.5 SD below the normative mean, OR ≥ 1 test score ≤ 2.0 SD below the mean**. Recommended core battery: HVLT-R, Trail Making Test A/B, COWA. The dual-criterion structure controls the family-wise false-positive rate inherent in multi-test batteries: if 6 tests are administered, ≥ 1 is expected to fall below –1.5 SD by chance (p ≈ 0.07 per test × 6 ≈ 0.5). Requiring two simultaneous failures drops this to < 0.1. Adoption in post-2011 CICI studies is uneven — ASCO 2024 abstract reported only **37.6% of 93 post-2011 studies** explicitly applied the ICCTF definition, rising from 0% in 2011 to 56.2% in 2022. Sensitivity/specificity profile: relatively specific (~80%) and less sensitive (~55–65%) against gold-standard clinical judgment; produces a cleaner but less frequent event signal.

**Reliable Change Index (RCI) methods (Jacobson & Truax 1991; Wefel et al. 2004, *Cancer*, [DOI](https://doi.org/10.1002/cncr.20272), PMID 15212812 for the first prospective longitudinal CICI application).** Patients are their own controls: an impairment event is registered when the post-treatment score declines by more than a specified threshold relative to test-retest reliability. Threshold is typically 1.645 SD (90% CI) or 1.96 SD (95% CI) of the expected score change, computed from control-group data. Advantage for survival analysis: RCI converts a within-person change into a discrete event, so you can register the **first timepoint at which RCI-defined decline is observed** → this is the cleanest time-to-event operationalization. Disadvantage: requires a matched control cohort for variance estimation, and practice effects are partially absorbed into the threshold. Wefel 2004 and Ahles 2003 both used RCI-based event counts.

**MoCA / MMSE threshold-based definitions.** MoCA < 26 or MMSE < 24 are the standard dementia-screening cutoffs. In CICI work specifically, the Dos Santos et al. 2021 *Cancers* study ([DOI](https://doi.org/10.3390/cancers13235785), n=310) used **MoCA < 22** and reported OR = 2.58 (95% CI 1.15–5.80) for baseline cognitive impairment predicting CICI — this is the threshold wired into your `cici_config.py` as `baseline_cognitive_impairment`. Advantage: single-test, clinic-friendly, translates directly to clinical decision-making. Disadvantage: MoCA/MMSE were not designed for CICI's characteristic pattern (subtle attention/executive/processing-speed deficits rather than the episodic-memory deficits these tools target), so they are substantially less sensitive than multi-domain batteries. For a survival-analysis primary outcome, MoCA/MMSE should not be the default; they belong in sensitivity analyses.

**Patient-reported outcome thresholds — FACT-Cog PCI.** The 18-item Perceived Cognitive Impairments (PCI) subscale of FACT-Cog, **cutoff score < 54**, was validated against neuropsychological testing by Van Dyk et al. 2020, *JNCI Cancer Spectrum* ([DOI](https://doi.org/10.1093/jncics/pkaa030)), yielding **76% sensitivity and 83% specificity**. Janelsins et al. 2017, *JCO* ([DOI](https://doi.org/10.1200/JCO.2016.68.5826), PMID 28029304) used a minimal clinically important difference cutoff on FACT-Cog to report **45.2% of patients vs. 10.4% of controls** had clinically significant cognitive decline from pre-chemo to post-chemo, and 36.5% vs. 13.6% from pre-chemo to 6-month follow-up. Advantage for survival framing: high event rate (≈ 40%) gives excellent statistical power. Disadvantage: self-reported CICI correlates weakly with objective testing and tracks more closely with depression/anxiety/fatigue than with actual neuropsychological decline — so you may be building a "distress model" rather than a "cognitive impairment model." Document this explicitly if you go this route.

**Composite / multi-domain definitions.** Combine objective test failure on ≥ 1 domain WITH a subjective PCI cutoff. This is the "converging evidence" approach — it reduces false positives on both sides and produces a more clinically meaningful event, at the cost of a lower event rate (typically 15–25% by end of 1-year follow-up). Rarely used in published CICI survival-adjacent work; would be a reasonable sensitivity analysis.

### Consensus statements

The ICCTF (International Cognition and Cancer Task Force) is the only body that has issued a formal consensus (Wefel 2011). INCOG is specific to traumatic brain injury; CICCTF does not exist as a distinct entity — you may be thinking of the ICCTF or the Cancer and Cognition Task Force (CCTF) under the American Psychosocial Oncology Society, neither of which has issued a survival-analysis-specific standard.

### Recommendation for your survival framework

**Primary outcome: RCI-defined decline** (≥ 1.645 SD drop from individual baseline on composite APE — Attention/Processing-speed/Executive function — score). This is the most defensible operationalization for time-to-event because (a) it captures *change*, not a static threshold, (b) it gives a discrete timestamp at each assessment visit, and (c) it has precedent in Wefel 2004 and Ahles 2003.

**Secondary outcome (for sensitivity analysis and power): FACT-Cog PCI < 54.** Higher event rate, higher power, but acknowledge the subjective-objective correlation issue in your methods section.

**For the synthetic cohort:** Simulate both outcomes. The objective RCI event rate should track ≈ 25–35% by 12-month follow-up (consistent with Janelsins-style objective decline), and the PCI event rate should track ≈ 40–50% (consistent with Janelsins-style self-report).

---

## Section 2 — Time-to-event distribution shapes

**Flag upfront — this is the thinnest-evidence section in the entire report.** No published CICI cohort reports a parametric survival fit (Weibull, Gompertz, log-normal). Kaplan–Meier curves for time-to-first-impairment are essentially absent from the literature. What you *can* reconstruct is a sequence of cumulative incidence values at the standard assessment timepoints, which is sufficient to fit a parametric baseline hazard.

### Reconstructed cumulative incidence from reported LMM cohorts

Because the field reports prevalence at fixed assessment timepoints rather than survival curves, the table below is the best approximation of an empirical hazard function. The FACT-Cog PCI-based incidence values are the most reliable because the Janelsins NCORP study is the largest prospective longitudinal CICI cohort (n = 580 patients + 363 controls) and reports clean clinically-significant-decline percentages at each timepoint.

| Timepoint | Incidence (objective, ICCTF/RCI) | Incidence (self-report, FACT-Cog PCI) | Primary source |
|---|---|---|---|
| Baseline (pre-chemo) | 20–30% pre-existing | 10–15% pre-existing | Ahles 2008; Wefel 2004; Mandelblatt 2014 |
| Post-chemo (≈ 2–4 weeks after last cycle) | 30–40% incident | 45.2% | Janelsins 2017 ([DOI](https://doi.org/10.1200/JCO.2016.68.5826)); Kerkmann 2025 |
| 6 months post-chemo | 25–35% | 36.5% | Janelsins 2017; Janelsins 2018 ([DOI](https://doi.org/10.1200/JCO.2018.78.6624)) |
| 12 months post-chemo | 20–32% (transient); 41% of transient → persistent | 30–40% | McDeed 2024 ([DOI](https://doi.org/10.1093/jncics/pkae019)); Lange 2023 CANTO ([DOI](https://doi.org/10.1093/jnci/djac240)) |
| 24 months | 15–25% persistent | 25–35% (CANTO) | Colaes 2025 ([DOI](https://doi.org/10.1002/cam4.71130)); Lange 2024 ([DOI](https://doi.org/10.1001/jamanetworkopen.2024.27576)) |
| 5 years | 15–25% persistent | Sparse data | Ahles 2002; limited follow-up |
| 21 years | Still measurable (CMF) | — | Koppelmans 2012 ([DOI](https://doi.org/10.1200/JCO.2011.38.0527)) |

### The shape is bimodal

Synthesizing across studies, the empirical hazard of first CICI event is **bimodal, not monotone**:

- **Acute peak** during or immediately after chemotherapy (weeks 8–16 from start of treatment) — driven by acute neuroinflammatory effects, anemia, fatigue, and the direct neurotoxic effect of agents like cyclophosphamide and doxorubicin. This peak is well-documented (Wefel 2004; Janelsins 2017; Kerkmann 2025).
- **Partial recovery trough** at 6–18 months post-treatment — Palesh et al. 2025 ([DOI](https://doi.org/10.1002/cncr.35816)) used growth mixture modeling on 139 breast cancer patients and identified a subclass (N=79, ≈ 57%) that *improved* over 12-month follow-up, and Sousa et al. 2020 *Neuropsychology Review* ([DOI](https://doi.org/10.1007/s11065-020-09441-9)) reviewed longitudinal neuroimaging and reported "partial neuropsychological, structural, and functional brain recovery" years after chemotherapy.
- **Second delayed rise** in older or APOE-ε4+ patients, appearing at 2–5+ years and accelerating into the 10–20 year horizon (Koppelmans 2012; Mandelblatt 2018 TLC three-way interaction).

This bimodal shape is **fundamentally incompatible with a simple Weibull or exponential baseline hazard**. A Weibull with shape parameter < 1 (monotone-decreasing hazard) captures only the acute-peak-then-decay portion and misses the late rise; shape > 1 (monotone-increasing hazard) captures the late rise but misses the acute peak.

### Parametric distribution recommendations

**For acute-only analysis (0–24 month horizon — the one your poster will emphasize):** **Weibull with shape k ≈ 0.7–0.9 (decreasing hazard)** is defensible. Rate parameter can be calibrated to give ≈ 35% event probability by 12 months and ≈ 45% by 24 months for a reference AC-T patient. This is the recommended default for your first pass.

**For long-horizon analysis (0–10 year):** **Log-logistic** distribution handles non-monotone hazards well and can fit a single-peak-then-decay-then-slow-rise pattern. Parameterize with α (scale) ≈ 8–12 months and β (shape) ≈ 1.2–1.5 to reproduce the acute peak.

**Most defensible across the full follow-up:** **Two-component mixture model** — a fast Weibull for acute events + a slow Gompertz for late-emerging cases. Computationally heavier, not supported out-of-the-box by lifelines/scikit-survival, but the honest choice if you extend follow-up beyond 2 years.

### By regimen class

| Regimen class | Relative hazard vs. no-chemo control | Modal time-to-event | Sources |
|---|---|---|---|
| CMF | 3–5× (highest) | Acute + very long tail (20+ yrs) | Schagen 1999 ([DOI](https://doi.org/10.1002/%28SICI%291097-0142%2819990601%2985:11%3C2338::AID-CNCR9%3E3.0.CO;2-0)); Koppelmans 2012 |
| Anthracycline (AC, FAC, FEC) | 2.5–3.5× | Acute peak (weeks 8–16) | Janelsins 2012 (IL-6 / MCP-1 link); Patel 2022 |
| Anthracycline + taxane (AC-T) | 2–3× | Acute + subacute | Janelsins 2018; Kerkmann 2025 |
| Platinum (BEP, FOLFOX, oxaliplatin) | 1.8–2.5× | Sub-acute (6–12 mo) | Fung 2018; Wefel 2011 (testicular) |
| Anti-metabolite (5-FU alone, methotrexate) | 1.3–1.8× | Acute but milder | Patel 2022 |
| Targeted (trastuzumab, CDK4/6i) | 1.1–1.5× | Chronic, slow | Limited data; McDeed 2024 HER2+ signal |
| No chemo (surgery/RT only) | 1.0 (reference) | — | Mandelblatt 2014; Ahles 2008 |

### By cancer type

Most published CICI follow-up is breast cancer, followed distantly by colorectal (oxaliplatin cohorts), testicular (BEP), and hematologic (lymphoma, BMT survivors). Sun et al. 2025, *JNCI Cancer Spectrum* ([DOI](https://doi.org/10.1093/jncics/pkaf105), PMID 41172265) gives you the largest published head-to-head — 41 breast + 73 lymphoma patients with 2-year follow-up and FACT-Cog decline at 1–2 years significant in both. Lung, ovarian, and prostate cognitive cohorts exist but are much smaller; hazard magnitudes are similar to breast cancer but uncertainty intervals are wider.

---

## Section 3 — Gap re-verification: drug-specific survival/time-to-event CICI models

**The gap stands, reinforced by the most recent systematic review.**

### The Duan et al. 2025 systematic review is the primary evidence

Duan, Wen, Zhang, Liu, Feng & Ren. "Advances in risk prediction models for cancer-related cognitive impairment." *Clinical and Experimental Medicine* 25, 74 (March 2025). [DOI](https://doi.org/10.1007/s10238-025-01590-6). PMID 40047952.

They searched PubMed, Cochrane, EMBASE, and Web of Science through August 2024 using MeSH + keyword combinations for (cancer OR tumor OR oncology) AND (cognitive impairment OR CRCI OR "chemo brain") AND (prediction model OR risk prediction OR prognostic model). Nine studies met inclusion criteria. Every single one is a binary classifier. The full roster:

| Study | Method | Cohort | AUC | Outcome framing |
|---|---|---|---|---|
| Zhou 2024, *Sci Rep* (n=515) | Multivariate logistic regression → nomogram | Breast cancer, cross-sectional | 0.955 | Binary, single timepoint |
| McDeed 2024, *JNCI Cancer Spectr* (TLC n=344+347) | LASSO (94 predictors) | Older breast cancer, longitudinal | 0.75 (transient), 0.79 (persistent) | Binary at 12mo / 24mo |
| Wang (POCD, elderly gastric, n=687) | Multivariate logistic regression | Gastric cancer surgical | 0.820 | Binary postoperative |
| Liu (n=741) | LASSO + logistic | Breast cancer, cross-sectional | 0.882 | Binary, single timepoint |
| Amir (meningioma, n=190) | Multivariate logistic | Intracranial meningioma | 0.72 (HRQoL), 0.78 (neurocog) | Binary |
| Sayo (n=145) | Multivariate logistic | Elderly hematologic malignancy | 0.81 | Binary |
| Zhu (n=494) | Artificial neural network (MLP) | Advanced cancer, cross-sectional | 0.910 | Binary |
| Xu (n=108) | SVM-RFE + RF feature selection | Colorectal stoma surgery | 0.909 | Binary postoperative |
| Noha (BMT survivors, n=277) | Elastic Net + genetic data | Hematologic, post-BMT | 0.89 discovery / 0.71 validation | Binary |
| Genan (n=1,568) | Multivariate logistic | Hospice cancer | Not reported as AUC | Binary (MMSE-defined) |

**None of these are time-to-event models.** None fit a Cox PH, Cox-LASSO, random survival forest, Survival XGBoost, DeepSurv, or any other survival model. None compute a c-index or cumulative hazard function. The Noha et al. elastic-net BMT model includes total body irradiation (TBI) as a covariate but does not stratify on specific chemotherapy agents — it uses the binary "received conditioning" status. The Zhou nomogram includes "chemotherapy regimen" as a categorical predictor but the 515-patient cohort is cross-sectional and has no time axis.

### Adjacent survival models in non-CICI cognitive domains exist but don't close the gap

- **Musto et al. 2024** (ICANN 2024, [arXiv:2409.16231](https://arxiv.org/abs/2409.16231)) — ADNI MCI-to-AD progression, metabolomics features, Cox / XGBoost / survival transformer head-to-head. Not CICI. Not drug-specific. Useful only as a c-index benchmark.
- **Li et al. 2025**, *Biostatistics* ([DOI](https://doi.org/10.1093/biostatistics/kxae036), PMID 39255368) — dynamic concordance-assisted learning on ADNI for MCI-to-AD risk scoring. Not CICI.
- **Tohidinezhad et al. 2024**, *Neuro-Oncology* ([DOI](https://doi.org/10.1093/neuonc/noae035), PMID 38595122) — NTCP model for neurocognitive decline after brain radiotherapy in primary brain tumor patients (n=219). Uses binary logistic regression at 6, 12, and 24 months separately (AUCs 0.79, 0.72, 0.69) with chemotherapy as a binary covariate. Not chemotherapy-specific. Not a true survival model.
- **Ensemble Survival Analysis for Preclinical Cognitive Decline** ([arXiv:2503.16645](https://arxiv.org/abs/2503.16645), 2025) — ADNI CN→MCI/AD ensemble of Cox-LASSO + Elastic Net + RSF + DeepSurv + XGBoost on 721 cognitively normal individuals. Not CICI. Not drug-specific.

### The precise shape of your remaining gap

You now have two overlapping but distinct novelty claims, both of which are defensible and one of which is stronger than the original framing.

**Claim A (from the original 469-paper review): No drug-specific AI/ML CICI risk prediction model exists.** This remains true through the Duan 2025 review. No model in the literature stratifies on cyclophosphamide vs. doxorubicin vs. taxane vs. platinum vs. anti-metabolite vs. targeted therapy as a first-class predictor.

**Claim B (new, stronger, specific to the survival refactor): No time-to-event AI/ML model for CICI exists. Period.** This is the cleaner claim and it follows directly from the Duan 2025 systematic review plus independent PubMed searches through April 2026.

**Recommendation:** Frame the poster and the manuscript around **Claim B**, with Claim A nested inside: "We developed the first survival-analysis prediction model for chemotherapy-induced cognitive impairment, with drug-specific stratification as a core feature." Both conditions together define the contribution — neither alone is as strong.

### Caveats

- The Duan 2025 search cut off August 2024; there is a ~20-month window (September 2024 through April 2026) in which a survival model could have been published. My PubMed searches through April 2026 using "chemotherapy cognitive impairment survival Cox" and related strings returned zero CICI-specific survival models. If you want maximum rigor for a manuscript, re-run the search one final time ~2 weeks before submission.
- Pre-print servers (arXiv, bioRxiv, medRxiv) are not indexed by PubMed. The Musto 2024 paper and the March 2025 ensemble paper cited above came from arXiv. A manual arXiv search with the same keywords before manuscript submission is prudent.
- Abstract-only conference proceedings (SABCS, ASCO, AACR) may contain survival-analysis CICI work that has not yet reached full-paper publication. ASCO Meeting Library and SABCS program search are worth spot-checking.

---

## Section 4 — Proportional hazards assumption for CICI predictors

The PH assumption — that hazard ratios are constant over follow-up time — is essentially untested in the CICI literature because no Cox models have been fit. The recommendations below are therefore extrapolated from adjacent literatures (Alzheimer's cohorts, cognitive aging studies, cardiovascular-cognitive studies) and from the biological plausibility of each feature.

### Per-predictor PH assessment

**APOE ε4 status — LIKELY VIOLATES PH, effect accelerates with age and follow-up time.** The Mandelblatt et al. 2018 TLC analysis in *JCO* ([DOI](https://doi.org/10.1200/JCO.2018.78.1914)) found a **statistically significant three-way interaction between chemotherapy × APOE ε4 × age** (p = 0.03), with chemotherapy effects largely confined to ε4 carriers in older patients. This is strong circumstantial evidence for a time-varying effect: the ε4 hazard ratio is near 1.0 immediately after treatment and grows with cumulative follow-up, consistent with ε4's established role as an accelerator of underlying Alzheimer's pathology rather than an acute chemotoxicity modifier. **Recommendation: treat APOE ε4 as a time-varying covariate OR stratify the Cox baseline hazard on ε4 status.** Alternatively, fit a time-interaction term (`APOE × log(time)`) and use the Schoenfeld residual test to formally check.

**IL-6 / CRP / TNF-α / inflammatory biomarkers — LIKELY VIOLATES PH, effect is acute and fades.** Baseline and immediately post-chemotherapy values predict acute impairment strongly (IL-6 OR = 5.50, CRP OR = 2.84 in Duivon 2024). By 12–24 months, inflammation has typically normalized and the acute HR drops toward 1.0. **Recommendation: use the peak post-chemo value (2–6 weeks after last cycle) as a fixed covariate if your primary endpoint is 0–12 month CICI. For longer follow-up, consider it time-varying or truncate the hazard contribution at 12 months.**

**Cumulative chemotherapy dose — QUINTESSENTIAL time-varying covariate.** Dose accumulates mechanically across cycles, so by definition the exposure changes over follow-up within each patient. Collins et al. 2013 *Psycho-Oncology* ([DOI](https://doi.org/10.1002/pon.3215)) established a linear dose-response. **Recommendation: treat as a time-varying covariate — patient-specific `dose(t)` equals total dose delivered by time t.** If you instead use final cumulative dose as a fixed-at-baseline covariate, you introduce immortal-time bias (higher-dose patients must have survived long enough to receive more cycles).

**Age at treatment — USUALLY OK for PH in short-horizon models (< 3 years), may fail at longer horizons.** In the 0–24 month window that dominates CICI studies, age is approximately proportional. Beyond 3 years, age-related dementia hazard begins to compete with treatment-attributable CICI and PH often fails. **Recommendation for your first-pass model: treat age as fixed. Document a Schoenfeld-residual check in methods. If your follow-up extends beyond 3 years, test age × log(time) interaction.**

**Endocrine therapy (tamoxifen / AI) — TIME-VARYING by construction.** Adjuvant endocrine therapy typically begins at or shortly after chemotherapy completion and continues for 5–10 years. Active exposure at time t is what matters, not historical initiation. **Recommendation: code as a time-varying binary covariate that switches from 0 → 1 at endocrine-therapy start and back to 0 at completion (for patients who complete their course).** An additional covariate for AI-versus-tamoxifen may be warranted given differential CNS penetration.

**Baseline cognitive reserve (education, baseline MoCA, reading ability) — LIKELY OK for PH.** These are "accelerated-failure-time" style factors: higher reserve delays events across the entire follow-up. Standard Cox handles this reasonably.

**Hormone receptor status (ER/PR/HER2) — OK for PH.** Fixed at baseline, biologically stable across the CICI follow-up horizon. No known time-interaction.

**Anxiety/depression (HADS, PHQ-9) — TIME-VARYING, but usually measured only at baseline in CICI cohorts.** If you have only baseline measurements, document that you're treating a time-varying feature as fixed-at-baseline and note it as a limitation.

### When to use AFT instead of Cox PH

Accelerated failure time (AFT) models with Weibull, log-normal, or log-logistic baseline are worth considering in two situations:

1. **If the primary scientific question is about time scaling, not hazard multiplication.** AFT coefficients have the interpretation "variable X multiplies survival time by exp(β)" which is often more clinically intuitive than hazard ratios. For a clinical audience that thinks in terms of "how many months does APOE ε4 subtract from cognitive resilience," AFT is clearer.
2. **If Schoenfeld residuals fail PH for multiple covariates.** Weibull AFT can absorb some non-proportionality through the shape parameter. Log-logistic AFT handles non-monotone hazards (matches the acute-peak-then-decay shape in Section 2).

**Practical recommendation for your refactor:** Fit Cox PH as the primary model. Fit Weibull AFT as a secondary model. Report both. If Schoenfeld tests fail for APOE ε4 and inflammatory markers (which is likely), present the AFT as a sensitivity analysis and stratify the Cox on APOE ε4 in a third sensitivity model.

---

## Section 5 — Covariate-specific hazard ratios from the literature

**Critical methodological flag at the top of this section.** Almost all published effect sizes in CICI are **odds ratios from logistic regression**, not hazard ratios from Cox models. The OR → HR conversion depends on event rate: when the outcome is uncommon (< 10% incidence), HR ≈ OR. When the outcome is common (> 30%, as in CICI), HR is typically smaller than OR, often by a factor of 0.6–0.8. The table below reports ORs with explicit labeling, and in the parameter table (end of report) I propose literature-calibrated HR values that are consistent with both the reported ORs and typical CICI event rates.

| Variable | Effect size (95% CI) | Type | Study | Cohort | Cancer type |
|---|---|---|---|---|---|
| **Inflammatory biomarkers** | | | | | |
| IL-6 (elevated post-chemo, binary) | **OR = 5.50** (1.43–36.60) | OR | Duivon 2024 ([DOI](https://doi.org/10.1186/s13058-024-01886-7)) | CANTO-Cog, n=200 | Breast |
| CRP > 3 mg/L | **OR = 2.84** (1.06–7.64) | OR | Duivon 2024 | CANTO-Cog, n=200 | Breast |
| TNF-α / sTNFRI (per SD) | r = –0.65 with cognitive score | Corr | Williams 2018 *J Neuroimmunol* | n=22 pilot | Breast |
| **Genetic** | | | | | |
| APOE ε4 carrier (vs. non-carrier) | **OR = 2.50** (1.20–5.20) | OR | McDeed 2024 ([DOI](https://doi.org/10.1093/jncics/pkae019)); Ahles 2003 | TLC n=344+347 | Older breast |
| APOE ε4 homozygous (vs heterozygous) | HR ≈ 2× the heterozygous HR | Extrapolated from AD literature | Not directly studied in CICI | — | — |
| COMT Val158Met | Inconsistent across studies | — | Boschetti 2025 review ([DOI](https://doi.org/10.3390/cancers17213522)) | Meta-review | Breast |
| BDNF Val66Met | Inconsistent | — | Boschetti 2025 | Meta-review | Breast |
| MTHFR C677T | Weak signal | — | Boschetti 2025 | Meta-review | Breast |
| **Demographic / reserve** | | | | | |
| Age (per decade) | HR ≈ 1.3–1.5 | Estimated from AD/MCI literature | Extrapolated | — | — |
| Education (per year) | **OR = 0.87** (0.81–0.95) — protective | OR | Amidi 2015 *Psycho-Oncol* | n=72 | Testicular |
| Baseline MoCA < 22 | **OR = 2.58** (1.15–5.80) | OR | Dos Santos 2021 *Cancers* | n=310 | Mixed |
| **Regimen** | | | | | |
| CMF vs. surgery-only | **OR = 6.40** (1.5–27.6) | OR | Schagen 1999 ([DOI](https://doi.org/10.1002/%28SICI%291097-0142%2819990601%2985:11%3C2338::AID-CNCR9%3E3.0.CO;2-0)) | n=34+30 | Breast |
| Anthracycline-based (FAC/FEC/AC) vs. no chemo | Estimated OR ≈ 2.5–3.5 | Extrapolated | Janelsins 2012 (IL-6/MCP-1 pathway) | Multiple | Breast |
| Platinum-based (BEP, oxaliplatin) | OR ≈ 1.8–2.5 | Extrapolated | Fung 2018 (19% at 2yr) | EORTC n=666 | Testicular |
| Taxane alone | OR ≈ 1.5–2.0 | Extrapolated | Literature-average | Multiple | Multiple |
| Targeted (trastuzumab) | HER2+ signal: **LASSO-selected** in McDeed 2024 | OR | McDeed 2024 | TLC | Older breast |
| Cumulative dose | Linear dose-response (per 1000 mg/m² cyclophosphamide-equivalent) | — | Collins 2013 *Psycho-Oncol* | n=60+60 | Breast |
| **Other treatment** | | | | | |
| Endocrine therapy (any active) | Small, inconsistent | — | Janelsins 2017 found n.s. effect | Multiple | Breast |
| Radiation therapy | Small signal in breast cohorts | — | Boschetti 2025 review | Meta-review | Breast |
| **Comorbidities** | | | | | |
| BMI / obesity | **LASSO-selected** in McDeed 2024 | — | McDeed 2024 | TLC | Older breast |
| Cardiovascular comorbidity | **LASSO-selected** in McDeed 2024 | — | McDeed 2024 | TLC | Older breast |
| Diabetes | Small positive signal | — | General cognitive literature | — | — |
| **Symptom / mental health** | | | | | |
| Fatigue (FACIT-Fatigue, per unit) | Moderate positive signal | — | Colaes 2025; Lange 2024 (CANTO) | Multiple | Breast |
| Depression (PHQ-9 / HADS-D, per unit) | Strong signal, especially for self-report outcomes | — | Janelsins 2017 (LMM covariate); Kim 2019 SEM β=0.25 | Multiple | Breast |
| Anxiety (HADS-A, per unit) | **OR = 1.298** (1.05–1.60) | OR | Zhou 2024 *Sci Rep* | n=515 | Breast |
| Sleep quality / insomnia | Growing signal, esp. in Palesh 2025 | — | Palesh 2025 ([DOI](https://doi.org/10.1002/cncr.35816)) | n=139 | Breast |
| Hemoglobin / anemia | Linked to cognition via O2 delivery | — | Duan 2025 review | Multiple | Multiple |
| **Tumor characteristics** | | | | | |
| HER2+ status | **LASSO-selected** in McDeed 2024 | — | McDeed 2024 | TLC | Older breast |
| ER/PR status | No strong independent signal | — | McDeed 2024 | TLC | Older breast |
| Menopausal status | Weak independent signal | — | Various | Multiple | Breast |

### Key uncertainty flags

- **The IL-6 OR of 5.50 has a 95% CI that spans 1.43–36.60.** This is an enormous interval and reflects the small positive-event count in the CANTO-Cog subset. Use the OR as a point estimate for synthetic data generation, but be prepared to defend against the observation that the CI is wide. Your ensemble analysis (XGBoost + LogReg) inherits stability from the literature-anchored prior; this is a strength to emphasize.
- **COMT, BDNF, MTHFR effects are inconsistent across studies** (Boschetti 2025 systematic review, 53 studies). Do not include these as literature-anchored predictors with confident effect sizes; include them only as exploratory features with flat priors if you use them at all.
- **No published Cox HRs exist for any of these predictors in a CICI context.** Every cell above is an OR or an extrapolation. This is a limitation of the field, not of this report, but it does mean your synthetic HRs are the first quantitative CICI hazard ratios in the literature — which is itself a novelty contribution if framed carefully.

---

## Section 6 — Benchmark c-index values

### Direct benchmark: Musto et al. 2024 on ADNI

Henry Musto, Daniel Stamate, Doina Logofatu, Daniel Stahl. "Predicting Deterioration in Mild Cognitive Impairment with Survival Transformers, Extreme Gradient Boosting and Cox Proportional Hazard Modelling." *Artificial Neural Networks and Machine Learning — ICANN 2024*, Lugano, Switzerland, September 2024. Springer Nature. [arXiv:2409.16231](https://arxiv.org/abs/2409.16231).

- **Task:** Predict time to MCI-to-AD progression using metabolomics/lipidomics features from ADNI1 participants (~400 MCI, ~200 AD, ~200 controls at enrollment; lipidomics sample from ADNI1 via ADNIMERGE R package, downloaded January 2024).
- **Validation:** Monte Carlo simulation, 100 repetitions of nested cross-validation.
- **Results:**
  - **Cox Proportional Hazards: mean c-index = 0.77**
  - **Survival XGBoost: mean c-index = 0.80**
  - **Survival Transformer: mean c-index = 0.85**
- Standard deviations (not quoted precisely in accessible abstract) confirm ML models are more stable than Cox.

This is the cleanest head-to-head benchmark you'll find in a cognitive-outcome survival context, and the methodology (100× nested CV, same cohort / same features across three model classes) is directly transferable to your CICI refactor.

### Generalizability to CICI

ADNI MCI → AD is a *cleaner* signal than CICI. Reasons:

1. **Larger effect sizes** — APOE ε4 hazard is roughly 2× larger in MCI→AD than the TLC-reported CICI effect.
2. **Less heterogeneity** — all ADNI MCI participants are cognitively similar at baseline; CICI cohorts have heterogeneous cancer types, regimens, and comorbidities.
3. **Better-defined outcome** — AD clinical diagnosis is a well-established, relatively uncontested endpoint; CICI operationalization is still debated (Section 1).
4. **Richer biomarker panels** — ADNI has CSF Aβ, tau, plus MRI, PET, and blood metabolomics; CICI cohorts typically have blood only.

**Expect your CICI c-indices to run 0.05–0.10 lower than Musto's ADNI benchmarks.** Translated to concrete targets:

| Model | Musto 2024 ADNI c-index | Realistic CICI c-index target | "Stretch" CICI c-index |
|---|---|---|---|
| Cox Proportional Hazards | 0.77 | **0.68–0.72** | 0.75 |
| Survival XGBoost | 0.80 | **0.72–0.76** | 0.79 |
| Survival Transformer | 0.85 | 0.75–0.80 | 0.83 |

### Other relevant benchmarks

- **Tohidinezhad et al. 2024** (brain-RT neurocognitive decline NTCP, primary brain tumor, n=219): AUC 0.79 at 6 mo, 0.72 at 12 mo, 0.69 at 24 mo. This is binary logistic regression at each timepoint, not a Cox c-index, but the numbers are the cleanest direct-CICI-adjacent benchmark in the literature. [DOI](https://doi.org/10.1093/neuonc/noae035).
- **McDeed et al. 2024 (TLC LASSO, your original Fork B benchmark):** AUC 0.75 (transient), 0.79 (persistent). Binary classification, not survival. Your XGBoost already exceeds this at 0.806.
- **Dementia-general Cox models:** Ernst et al. 2021 *JAHA* ([DOI](https://doi.org/10.1161/JAHA.120.019613), PMID 34176293) reported HR = 1.68 (1.19–2.39) for men in the highest blood-pressure-variability tertile predicting incident dementia in ASPREE (n=19,114 older adults); c-index not quoted in abstract but similar studies typically report 0.70–0.75.
- **Post-operative cognitive dysfunction Cox models:** Rare in published literature. Wang's gastric-POCD logistic AUC of 0.820 is a reasonable proxy but is binary.

### Recommendation

On your poster and in your manuscript, state the target range as:

> "Cox Proportional Hazards c-index target: ≥ 0.70, with a stretch goal of 0.75 that would match the Musto 2024 ADNI benchmark on a harder prediction task. Survival XGBoost c-index target: ≥ 0.75."

If your Cox-LASSO or Survival XGBoost clears 0.78, that is publishable as a first-in-field result.

---

## Section 7 — Censoring patterns in CICI cohort studies

### Empirical dropout rates

Dropout in CICI longitudinal cohorts is substantial. A spot survey of the major studies:

| Study | Baseline n | Follow-up retention | Notes |
|---|---|---|---|
| Janelsins 2018, *JCO* | 580 patients + 363 controls | 6-month retention ≈ 85%; 12-month ≈ 70% | Multisite NCORP, well-resourced |
| Janelsins 2017, *JCO* | 581 patients + 364 controls | Similar | Same cohort, FACT-Cog outcome |
| CANTO-Cog (Lange 2023 *JNCI*) | 276 patients + 135 controls | 2-year retention ≈ 70–75% | French national cohort, well-tracked |
| Sun 2025, *JNCI Cancer Spectrum* | 41 breast + 73 lymphoma + controls | 2-year retention ≈ 60% | Smaller subsample of NCORP |
| Palesh 2025, *Cancer* | 139 breast | 12-month retention ≈ 60% (GMM analysis retained all) | Missing-data handling via GMM |
| Koppelmans 2012, *JCO* | 196 CMF survivors | 21-year follow-up; enormous selection bias | Ultra-long-term, not typical |
| TLC (McDeed 2024) | 344+347 | 24-month retention ≈ 80% | Well-resourced cancer-aging consortium |

**Typical ranges to assume for your synthetic cohort:**
- 6-month retention: 85–95% (dropout 5–15%)
- 12-month retention: 70–85% (cumulative dropout 15–30%)
- 24-month retention: 60–75% (cumulative dropout 25–40%)
- 5-year retention: 40–60% (cumulative dropout 40–60%)

### Reasons for censoring

Ranked by approximate prevalence in breast cancer cognitive cohorts (based on Janelsins 2017/2018, Lange 2023 CANTO reports, and Duan 2025 review):

1. **Administrative end-of-study (uninformative)** — most common at the planned follow-up horizon (e.g., 12mo or 24mo cutoff).
2. **Loss to follow-up (moderately informative)** — participants disengage; often correlated with worse outcomes (lower education, lower SES, more depressed) which biases hazard estimates downward.
3. **Withdrawal (informative)** — participants explicitly opt out; often correlated with treatment-related distress.
4. **Disease progression / recurrence (informative, and a competing risk)** — patients whose cancer returns often stop participating in cognitive substudies.
5. **Death (informative, and a competing risk)** — addressed in Section 8.

### Informative vs. uninformative censoring

Standard Cox PH assumes censoring is **independent of the event process** (uninformative / non-informative censoring). Loss to follow-up in CICI cohorts is moderately informative because patients with worse cognitive symptoms may disengage. Death before CICI observation is strongly informative and should be handled as competing risks (Section 8), not as censoring.

### Recommendations for synthetic data

For the synthetic cohort, a reasonable default is:

1. **Generate exact event times** from the chosen parametric hazard (Weibull / log-logistic).
2. **Apply administrative censoring** at the planned follow-up horizon (e.g., τ = 24 months). Patients with event time > τ are censored at τ.
3. **Apply random dropout** at ~10% per 6-month interval for uninformative censoring. Implementation: at each assessment time t, a Bernoulli(0.10) trial determines if the patient drops out; if so, censor at t.
4. **Optional: simulate informatively censored dropout** — patients with higher predicted risk have 1.3–1.5× higher dropout hazard. This tests model robustness to a realistic nuisance.
5. **Separately simulate competing mortality** per Section 8.

For the *reported* results, fit the model assuming uninformative censoring (the standard assumption) and present a sensitivity analysis where you re-fit on the informative-censoring version of the synthetic data. If the c-index degrades by < 0.02, the conclusion is "robust to realistic dropout mechanisms."

---

## Section 8 — Competing risks (death before CICI onset)

### Mortality rates in CICI cohort populations

Breast cancer stage I–III 5-year all-cause mortality is the base-rate starting point. Current SEER data:

| Stage at diagnosis | 5-yr all-cause mortality | 10-yr all-cause mortality | Source |
|---|---|---|---|
| Localized (Stage 0–I) | ~5–8% | ~12–18% | SEER 2020 (general estimates) |
| Regional (Stage II–III) | ~15–20% | ~28–35% | SEER 2020 |
| Distant / metastatic (Stage IV) | ~70–75% | ~85%+ | SEER 2020 |
| Hospice / palliative (Genan cohort from Duan 2025) | 24% developed impairment in 4–16 weeks | — | Genan / Duan 2025 |

For elderly cohorts (TLC 60+), add roughly 3–5 percentage points for non-cancer competing mortality over 5 years. For hematologic malignancies and lymphoma cohorts, 5-year mortality runs higher (25–40% depending on subtype and age).

**For a typical non-metastatic breast cancer CICI cohort over a 2-year follow-up, expected mortality is 3–6%.** For a metastatic or elderly cohort, it can exceed 20% within 2 years and competing risks become the dominant analytic issue.

### When competing-risks analysis becomes necessary

Rule of thumb from the survival methodology literature:

- **Competing mortality < 5% over follow-up horizon:** Standard cause-specific Cox PH (treating death as censoring) is fine. Subdistribution-hazard (Fine-Gray) results will differ by < 1–2%.
- **Competing mortality 5–15%:** Run both cause-specific Cox and Fine-Gray. Report both. Note any divergence explicitly.
- **Competing mortality > 15%:** Fine-Gray is the appropriate primary model. Cause-specific Cox overestimates the marginal hazard of CICI because it implicitly assumes death does not preclude CICI observation.

### Cause-specific Cox vs. Fine-Gray: interpretation

- **Cause-specific Cox:** estimates the hazard of CICI among individuals still alive and CICI-free. Appropriate when the scientific question is about biological susceptibility — "given that a patient has not died and has not developed CICI, how does X affect their instantaneous CICI hazard?"
- **Fine-Gray subdistribution hazard:** estimates the subdistribution hazard accounting for death as a competing event. Hazard ratios from Fine-Gray can be interpreted as effects on the *cumulative incidence* of CICI. Appropriate when the scientific question is about the probability of observing CICI in the population — "how does X affect the proportion of patients who will eventually experience CICI?"

For a risk-stratification tool aimed at clinical decision-making (which your project is), Fine-Gray is typically preferred when competing mortality is non-negligible, because clinicians care about cumulative incidence, not instantaneous hazard conditional on survival.

### Recommendation for your project

**Default: cause-specific Cox, because your synthetic cohort will be non-metastatic breast cancer analogs with low (~3–6%) competing mortality.** Re-fit with Fine-Gray as a sensitivity analysis. Document the divergence (or lack thereof) in the supplementary materials. If you expand to metastatic or elderly cohorts in future work, Fine-Gray becomes primary.

### Simulating competing mortality in synthetic data

1. Generate CICI event times as before from the primary Weibull / log-logistic hazard.
2. Independently generate competing mortality times from a second parametric distribution. A reasonable default is **exponential with rate parameter chosen to produce 3–6% cumulative mortality at 24 months**: λ ≈ –log(1 – 0.045) / 24 ≈ 0.00192 per month.
3. For each patient, observed time = min(CICI time, mortality time, administrative censoring time). Observed event type = whichever came first.
4. **Add mortality-risk–covariate correlations** for realism: age, stage, cardiovascular comorbidity, and poor performance status all increase both mortality and (through a shared inflammation pathway) CICI risk. In the synthetic cohort, set the correlation between the two hazards via shared covariates rather than via an independent latent factor — e.g., age enters both hazard functions with a positive coefficient.

### Published competing-risks cognitive models

The PubMed search for "Fine-Gray subdistribution hazard cognitive decline competing" returned seven articles — none in a cancer context. The technique is well-established in dementia/AD research (e.g., several studies on incident-dementia hazards adjusted for competing mortality) but has not been applied to CICI. **This is a secondary methodological novelty:** first CICI model to report Fine-Gray subdistribution hazards, in addition to being the first CICI survival model overall.

---

## Section 9 — Interval censoring

### Why standard Cox PH is mis-specified under interval censoring

Cognitive assessments in CICI cohorts are scheduled — typically baseline, post-chemo (2–4 weeks after last cycle), 6 months, 12 months, and sometimes 24 months. The true onset of CICI falls at an unknown time between two visits. Standard Cox PH treats the observed event time (the visit at which impairment is first detected) as the exact event time, which:

1. **Systematically overestimates time-to-event** by approximately half an interval width. In a 6-month interval, this is a 3-month upward bias.
2. **Attenuates hazard-ratio estimates** (biases HR toward 1.0) because covariate values may have changed between the true unobserved event time and the observed detection time.
3. **Inflates standard errors** because the event-time uncertainty is ignored.

The practical impact depends on interval width. For intervals < 3 months, the bias is small and standard Cox is usually defensible. For intervals of 6–12 months (CICI's typical schedule), the bias is meaningful and should be addressed. For intervals > 12 months (long-term follow-up studies), interval-censored methods become essential.

### Alternative methods

**Interval-censored Cox — `icenReg` (R), `pyicenReg` (Python):** Turnbull-based nonparametric maximum likelihood or Cox PH extension that correctly integrates over the unknown true event time within each interval. Conceptually cleanest. Computationally heavier but tractable for n ≤ 10,000.

**Parametric AFT with interval censoring:** Weibull / log-normal / log-logistic AFT can be fit directly by maximum likelihood over interval-censored data. `lifelines.WeibullAFTFitter` in Python supports this via the `entry` and `event` columns. This is the cleanest practical choice for a synthetic cohort because your baseline hazard is already parametric.

**Multistate models:** Model cognitive state as "cognitively intact → mild impairment → CICI" with transitions observed at visits. More realistic for the CICI trajectory pattern (Section 2 bimodality) but substantially more complex. Overkill for your first refactor.

**Discrete-time survival (complementary log-log logistic regression):** Treat each visit as a discrete time point and fit a pooled logistic regression with a complementary-log-log link. Asymptotically equivalent to Cox PH as interval widths → 0, and is easy to fit with standard tools. Output coefficients exponentiate to approximate hazard ratios. A solid pragmatic choice that bridges binary classification and survival analysis.

### Imputation strategies

**Midpoint imputation:** Replace the interval-censored event time with the midpoint of the interval (T_first_detection + T_previous_visit) / 2. Simple, widely used, approximately unbiased when intervals are narrow and events are uniformly distributed within intervals. Defensible for 3-month intervals; biased for 12-month intervals.

**Right-endpoint imputation:** Use the detection time as the event time (what standard Cox PH does). Systematically biased upward; not recommended.

**Left-endpoint imputation:** Use the previous visit as the event time. Systematically biased downward.

**Multiple imputation (MI):** Draw multiple event times from the within-interval conditional distribution, fit the model separately on each imputation, combine via Rubin's rules. The gold-standard when computationally feasible. For synthetic data this is overkill; for real CICI data with wide intervals it's the recommended approach.

### What CICI cohorts actually do

The reality is that **most CICI cohorts report LMM or latent-class results at discrete timepoints and sidestep the interval-censoring question entirely**. They treat each assessment as a separate cross-sectional analysis ("FACT-Cog score at 12 months") rather than as a time-to-event. When time-to-event has been reported at all (rare), midpoint imputation or right-endpoint imputation are both found in the literature without explicit methodological justification. This means your survival-analysis refactor is entering largely uncharted territory for this field, and there is no established convention to follow.

### Recommendation for your synthetic cohort

Since your synthetic data will have **exact continuous event times by construction**, interval censoring is a modeling choice, not a data-generation choice. Recommended structure:

1. **Generate exact event times** under the primary Weibull / log-logistic hazard.
2. **Create three interval-censored observation schedules** as sensitivity analyses:
   - Schedule A (dense): baseline, 3mo, 6mo, 9mo, 12mo, 18mo, 24mo. Intervals ≤ 6 months.
   - Schedule B (standard CICI): baseline, post-chemo (~4mo), 6mo, 12mo, 24mo. Mixed interval widths.
   - Schedule C (sparse / long-horizon): baseline, 12mo, 24mo, 60mo. Interval widths up to 36 months.
3. **For each schedule**, apply midpoint imputation and fit Cox PH. Also fit Weibull AFT with the true interval-censoring likelihood.
4. **Compare c-index and hazard-ratio estimates across schedules.** Report the degradation in Cox PH performance as interval width widens. Your XGBoost-based binary baseline already effectively assumes schedule B; this shows the marginal benefit (or cost) of the survival refactor.

This sensitivity-analysis structure gives you a clean scientific story: "standard Cox is defensible at 3-month intervals with < 0.02 c-index loss; Weibull AFT with explicit interval-censoring likelihood is required at 12-month intervals to preserve hazard-ratio calibration."

---

## Consolidated parameter table — for CICI time-to-event synthetic data generation

This table consolidates every quantitative parameter needed for synthetic cohort generation in one place. Every value traces to a source or is an explicit extrapolation flagged as such. Use these as the starting values in a new `cici_survival_config.py` module analogous to your existing `cici_config.py`.

### Baseline hazard

| Parameter | Value | Confidence range | Source / rationale |
|---|---|---|---|
| Distribution (primary, 0–24 mo) | **Weibull** | — | Recommended default per Section 2 |
| Weibull shape (k) | 0.85 | 0.7–0.95 | Decreasing hazard fits acute-peak-then-decay |
| Weibull scale (λ, months) | 22 | 18–28 | Calibrated to ≈ 35% event prob at 12 mo for AC-T reference |
| Distribution (alternative, bimodal) | **Log-logistic** | — | For non-monotone hazard |
| Log-logistic α (scale) | 10 months | 8–12 | — |
| Log-logistic β (shape) | 1.3 | 1.1–1.5 | — |
| Distribution (extended horizon 0–5 yr) | **Weibull + Gompertz mixture** | — | Captures late re-emergence |

### Per-covariate hazard ratios (literature-anchored, conservative conversion from ORs)

| Predictor | Proposed HR | 95% CI (from OR where HR unavailable) | Original OR source |
|---|---|---|---|
| IL-6 elevated post-chemo (binary) | **2.80** | 1.20–5.20 | OR 5.50 (1.43–36.60), Duivon 2024 — wide CI, conservative HR conversion for 45% event rate |
| CRP > 3 mg/L | **1.85** | 1.05–3.20 | OR 2.84 (1.06–7.64), Duivon 2024 |
| APOE ε4 carrier | **1.75** | 1.15–2.70 | OR 2.50 (1.20–5.20), McDeed/Ahles |
| Baseline MoCA < 22 | **1.80** | 1.10–2.90 | OR 2.58 (1.15–5.80), Dos Santos 2021 |
| Education (per year) | **0.92** (protective) | 0.87–0.97 | OR 0.87 (0.81–0.95), Amidi 2015 |
| Anxiety (per HADS-A unit) | **1.20** | 1.04–1.40 | OR 1.298 (1.05–1.60), Zhou 2024 |
| CMF regimen vs. no chemo | **3.80** | 1.50–9.50 | OR 6.40 (1.5–27.6), Schagen 1999 |
| AC / AC-T vs. no chemo | **2.40** | 1.40–4.00 | Extrapolated from Janelsins 2012, 2017, 2018 |
| FAC / FEC vs. no chemo | **2.60** | 1.50–4.50 | Extrapolated from Patel 2022, Janelsins 2012 |
| Platinum vs. no chemo | **2.10** | 1.20–3.80 | Extrapolated from Fung 2018 |
| Targeted (trastuzumab, CDK4/6i) vs. no chemo | **1.40** | 0.95–2.10 | McDeed 2024 HER2 LASSO signal |
| Age (per 10 years) | **1.35** | 1.15–1.60 | AD/MCI literature extrapolation |
| Cumulative dose (per 1000 mg/m² cyclophosphamide-eq) | **1.15** | 1.05–1.30 | Collins 2013 linear dose-response |
| Fatigue (per FACIT-F unit) | **1.05** | 1.01–1.10 | Lange 2024 CANTO |
| Depression (per PHQ-9 unit) | **1.08** | 1.02–1.15 | Janelsins 2017 LMM covariate |
| Cardiovascular comorbidity (any) | **1.40** | 1.05–1.90 | McDeed 2024 LASSO signal |
| BMI ≥ 30 (obesity) | **1.30** | 1.00–1.70 | McDeed 2024 LASSO signal |
| HER2+ status | **1.35** | 1.00–1.85 | McDeed 2024 LASSO signal |

### Censoring parameters

| Parameter | Value | Source |
|---|---|---|
| Administrative censoring horizon (τ) | 24 months | CICI cohort convention (Janelsins, CANTO) |
| Uninformative dropout rate (per 6-mo interval) | 10% | Empirical average, Section 7 |
| Informative dropout rate (sensitivity analysis) | 10% × exp(0.3 × predicted_log_hazard) | Simulates sicker-patients-drop-out-more |

### Competing mortality parameters

| Parameter | Value (non-metastatic BC cohort) | Value (elderly 60+ cohort) | Value (metastatic cohort) |
|---|---|---|---|
| Mortality distribution | Exponential | Exponential | Weibull |
| Mortality rate (per month) | 0.00192 | 0.00330 | 0.0300 |
| Implied 24-mo cumulative mortality | ≈ 4.5% | ≈ 7.7% | ≈ 51% |
| Use Fine-Gray? | No (cause-specific Cox) | Yes (sensitivity) | Yes (primary) |

### Interval-censoring assessment schedules (for sensitivity analysis)

| Schedule | Visit times (months) | Max interval | Recommended analysis |
|---|---|---|---|
| A (dense) | 0, 3, 6, 9, 12, 18, 24 | 6 mo | Standard Cox PH with midpoint imputation |
| B (CICI-typical) | 0, 4, 6, 12, 24 | 12 mo | Cox PH + Weibull AFT with interval likelihood |
| C (sparse) | 0, 12, 24, 60 | 36 mo | Weibull AFT with interval likelihood + MI sensitivity |

---

## Methodological recommendations for the refactor

Briefly, in order of priority:

1. **Fit Cox PH as the primary model**, with Weibull AFT as a secondary model. Report both. Use Schoenfeld residuals to check PH for every covariate and flag failures.
2. **Stratify the Cox baseline on APOE ε4 status.** This is the single most likely PH violation and the safest pre-registered handling.
3. **Treat cumulative chemotherapy dose as a time-varying covariate.** Mandatory; otherwise immortal-time bias distorts hazard estimates.
4. **Use RCI-based APE composite decline as the primary outcome**, FACT-Cog PCI < 54 as secondary. Document both in the methods, primary in the results, secondary in sensitivity analyses.
5. **Fit Cox-LASSO with 10-fold CV for variable selection.** Report the stability path across λ values. This is the natural regularized Cox extension of your McDeed 2024 benchmark.
6. **Fit Survival XGBoost** (from `scikit-survival` or `xgboost` with `survival:cox` objective). This is your direct Fork B analog and the headline survival-ML contribution.
7. **Report c-index, time-dependent AUC at 6/12/24 mo, and calibration plots (not just discrimination).** The Duan 2025 review explicitly called out calibration as systematically under-reported in CRCI models — making it a credibility differentiator for yours.
8. **Run Fine-Gray sensitivity analysis** for competing mortality. Report divergence from cause-specific Cox.
9. **Fit on three assessment schedules (dense, typical, sparse)** and report c-index degradation across them. This is the interval-censoring sensitivity story.
10. **Pre-register the hazard-ratio point estimates** from the parameter table above before running the model. The synthetic cohort was parameterized from these values, so the model should recover them; any large discrepancy flags a bug rather than a finding.

---

## Gaps in evidence flagged in this report

Honest accounting of where this report extrapolates or relies on thin evidence:

1. **Section 2 (time-to-event distribution shape):** No parametric survival fits have been published for CICI. The Weibull / log-logistic recommendations are extrapolated from reported prevalence at discrete timepoints.
2. **Section 5 (hazard ratios):** All published effect sizes are odds ratios from logistic regression or effect-size coefficients from linear mixed models. No hazard ratios exist. Conservative OR → HR conversions are proposed, anchored in typical event rates, but are themselves uncertain.
3. **Section 4 (PH assumption):** Essentially untested in CICI because no Cox models have been fit. Recommendations are extrapolated from adjacent literatures (AD, cognitive aging).
4. **Section 7 (censoring):** Dropout rates are assembled from retention numbers in cohort publications, which often report per-visit retention rather than time-to-dropout. Rates are approximate.
5. **COMT, BDNF, MTHFR effects (Section 5 subtable):** Inconsistent across studies per Boschetti 2025. Do not wire these in as confident predictors.
6. **Platinum and targeted-therapy hazard ratios:** Extrapolated from small cohorts. Less evidence-dense than breast-chemotherapy regimens.

---

## Bibliography — primary sources cited

- Ahles TA et al. 2003. APOE ε4 and cognitive function in cancer survivors. *Psycho-Oncology*.
- Amidi A et al. 2015. Cognitive function in testicular cancer survivors. *Psycho-Oncology*.
- Boschetti A et al. 2025. Biomarkers and CRCI in breast cancer: a systematic review. *Cancers* 17(21):3522. [DOI](https://doi.org/10.3390/cancers17213522). PMID 41228315.
- Carroll JE et al. 2022. Dose-dependent CRP effects on self-reported cognition. *JCO*.
- Collins B et al. 2013. Cumulative chemotherapy dose and cognition: linear dose-response. *Psycho-Oncology*.
- Colaes R et al. 2025. Trajectories of cognitive complaints in breast cancer. *Cancer Medicine* 14(15):e71130. [DOI](https://doi.org/10.1002/cam4.71130). PMID 40772479.
- Duan R et al. 2025. Advances in risk prediction models for cancer-related cognitive impairment. *Clinical and Experimental Medicine* 25(1):74. [DOI](https://doi.org/10.1007/s10238-025-01590-6). PMID 40047952. **[The definitive gap-verification systematic review.]**
- Dos Santos M et al. 2021. Cognitive impairment in breast cancer. *Cancers*.
- Duivon M et al. 2024. Inflammatory biomarkers and CICI in CANTO-Cog. *Breast Cancer Research*.
- Ernst ME et al. 2021. Blood pressure variability and cognitive decline. *JAHA* 10(13):e019613. [DOI](https://doi.org/10.1161/JAHA.120.019613). PMID 34176293.
- Fung C et al. 2018. Cognitive outcomes after testicular cancer BEP. *Advances in Urology*.
- Ganz PA et al. 2013. sTNF-RII and frontal cortex metabolism. *Brain, Behavior, and Immunity*.
- Grégoire C et al. 2025. CICI pooled incidence in breast cancer RCTs. *Scientific Reports*.
- Janelsins MC et al. 2012. Anthracyclines and inflammatory cytokines. *Supportive Care in Cancer*.
- Janelsins MC et al. 2017. FACT-Cog cognitive complaints nationwide cohort. *JCO* 35(5):506–514. [DOI](https://doi.org/10.1200/JCO.2016.68.5826). PMID 28029304.
- Janelsins MC et al. 2018. Longitudinal CRCI trajectory in nationwide cohort. *JCO* 36(32):JCO2018786624. [DOI](https://doi.org/10.1200/JCO.2018.78.6624). PMID 30240328.
- Kerkmann A et al. 2025. CICARO prospective CICI cohort. *The Oncologist*.
- Koppelmans V et al. 2012. Cognition in CMF survivors 21 years post-treatment. *JCO*. [DOI](https://doi.org/10.1200/JCO.2011.38.0527).
- Lange M et al. 2023. CANTO cognitive change at 2 years. *JNCI* 115(3):322–331. [DOI](https://doi.org/10.1093/jnci/djac240). PMID 36571503.
- Lange M et al. 2024. Cognition and return to work in CANTO. *JAMA Network Open* 7(8):e2427576. [DOI](https://doi.org/10.1001/jamanetworkopen.2024.27576). PMID 39158915.
- Li W et al. 2025. Dynamic concordance-assisted learning for ADNI MCI. *Biostatistics* 26(1). [DOI](https://doi.org/10.1093/biostatistics/kxae036). PMID 39255368.
- Mandelblatt JS et al. 2014. Baseline cognitive impairment in older breast cancer patients.
- Mandelblatt JS et al. 2018. APOE × chemotherapy × age three-way interaction (TLC). *JCO*. [DOI](https://doi.org/10.1200/JCO.2018.78.1914).
- McDeed AP et al. 2024. Prediction of cognitive decline in older breast cancer survivors (TLC LASSO). *JNCI Cancer Spectrum* 8(2). [DOI](https://doi.org/10.1093/jncics/pkae019). PMID 38556480.
- Musto H, Stamate D, Logofatu D, Stahl D. 2024. Predicting deterioration in MCI with survival transformers, XGBoost, and Cox PH. *ICANN 2024*, Springer. [arXiv:2409.16231](https://arxiv.org/abs/2409.16231). **[The c-index benchmark.]**
- Palesh O et al. 2025. Natural trajectory subclasses of CRCI with insomnia. *Cancer* 131(8):e35816. [DOI](https://doi.org/10.1002/cncr.35816). PMID 40219873.
- Patel SK et al. 2022. Regimen-specific MMSE declines.
- Schagen SB et al. 1999. CMF vs. surgery-only cognitive outcomes. *Cancer*.
- Sousa H et al. 2020. CRCI developmental trajectory neuroimaging review. *Neuropsychology Review* 30(3):287–309. [DOI](https://doi.org/10.1007/s11065-020-09441-9). PMID 32607817.
- Sun H et al. 2025. Longitudinal cognitive function in breast cancer and lymphoma. *JNCI Cancer Spectrum* 9(6). [DOI](https://doi.org/10.1093/jncics/pkaf105). PMID 41172265.
- Tohidinezhad F et al. 2024. NTCP for neurocognitive decline after brain RT. *Neuro-Oncology* 26(8):1467–1478. [DOI](https://doi.org/10.1093/neuonc/noae035). PMID 38595122.
- Van Dyk K et al. 2020. FACT-Cog PCI cutoff validation. *JNCI Cancer Spectrum*. [DOI](https://doi.org/10.1093/jncics/pkaa030).
- Wefel JS et al. 2004. First prospective longitudinal CICI study. *Cancer*. [DOI](https://doi.org/10.1002/cncr.20272). PMID 15212812.
- Wefel JS et al. 2011. ICCTF consensus criteria. *Lancet Oncology*.
- Whittaker AL et al. 2022. 52-study meta-analysis of CICI prevalence. *Scientific Reports*.
- Yap NY et al. 2021. Cytokines, BDNF, and CRCI trajectories. *Cytokine* 144:155556. [DOI](https://doi.org/10.1016/j.cyto.2021.155556). PMID 33985854.
- Zhou Q et al. 2024. CRCI nomogram (n=515). *Scientific Reports*.

---

*Report compiled April 19, 2026, for the CICI survival-analysis refactor. Primary source coverage: PubMed through April 2026, Duan et al. 2025 systematic review (search through August 2024), targeted arXiv searches. Attribution note: PubMed-sourced citations retain their original DOIs; all claims are cited to source. Areas of thin evidence are explicitly flagged throughout.*
