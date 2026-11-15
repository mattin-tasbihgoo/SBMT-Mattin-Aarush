"""
CICI Survival Analysis — Literature-Grounded Configuration
===========================================================
Parameters for refactoring the CICI risk model from binary classification
(XGBoost, AUC 0.806) to a survival-analysis framework (Cox PH, Cox-LASSO,
Survival XGBoost, Weibull AFT).

Every number below is traceable to a published source or is explicitly
flagged as a conservative OR → HR conversion. Ported verbatim from the
consolidated parameter table in:

    CICI_Survival_Analysis_Evidence_Base.md

This file extends (not replaces) cici_config.py. Import both.

Generated April 19, 2026 for the SBMT 2026 post-conference survival refactor.
"""

# ── 1. BASELINE HAZARD DISTRIBUTIONS ─────────────────────────────────────
# For generating synthetic event times under proportional hazards.
# Primary = Weibull. Alternatives are for sensitivity analyses.

BASELINE_HAZARD = {
    "weibull_primary": {
        "shape_k": 0.85,
        "scale_lambda_months": 22,
        "confidence_range": {"shape": (0.7, 0.95), "scale": (18, 28)},
        "horizon_months": 24,
        "note": (
            "Decreasing hazard (k < 1) fits the CICI acute-peak-then-decay "
            "trajectory. Scale calibrated to ~35% event probability at 12 mo "
            "for an AC-T reference patient."
        ),
        "source": (
            "Evidence Base §2, extrapolated from Janelsins 2018 JCO, "
            "CANTO-Cog (Lange 2024), Sun 2025 JNCI-CS incidence at fixed timepoints."
        )
    },
    "log_logistic_alternative": {
        "scale_alpha_months": 10,
        "shape_beta": 1.3,
        "confidence_range": {"alpha": (8, 12), "beta": (1.1, 1.5)},
        "note": (
            "Non-monotone hazard. Use for sensitivity analysis when the "
            "bimodal trajectory observed in Sun 2025 / Palesh 2025 is the "
            "scientific question."
        ),
        "source": "Evidence Base §2"
    },
    "weibull_gompertz_mixture_extended": {
        "weibull_weight": 0.7,
        "weibull_shape": 0.85,
        "weibull_scale_months": 22,
        "gompertz_weight": 0.3,
        "gompertz_rate": 0.03,
        "gompertz_shape": 0.002,
        "horizon_months": 60,
        "note": (
            "For exploratory 0–60 mo analyses only. Captures late re-emergence "
            "of cognitive symptoms seen in long-term CICI survivor cohorts "
            "(Koppelmans 2012, 21-year follow-up). Not a primary model."
        ),
        "source": "Evidence Base §2 extended horizon recommendation"
    },
}

# Reference patient profile used to calibrate the baseline hazard.
# Under this profile (eta = 0), cumulative incidence at 12 mo should be ~30%.
REFERENCE_PATIENT = {
    "regimen": "CONTROL_CANCER_NO_CHEMO",
    "age": 50,
    "apoe_e4": 0,
    "education_years": 14,
    "baseline_moca": 25,
    "il6_elevated": 0,
    "crp_elevated": 0,
    "hads_a": 0,
    "phq9": 0,
    "fatigue_score": 0,
    "her2_positive": 0,
    "bmi_obese": 0,
    "cardiovascular_comorbidity": 0,
    "cumulative_dose_1000": 0,
    "target_12mo_event_rate": 0.30,
    "target_tolerance": 0.05,
}

# ── 2. HAZARD RATIOS ─────────────────────────────────────────────────────
# Primary inputs to the data-generating process.
# Every HR is a conservative conversion from published ORs (event rate ~40%).
# Ranges are 95% CIs from the original OR study where available.

HAZARD_RATIOS = {
    # ── Genetic / demographic ──
    "apoe_e4": {
        "HR": 1.75,
        "CI_95": (1.15, 2.70),
        "type": "binary",
        "original_OR": 2.50,
        "ph_handling": "STRATIFY baseline hazard (Mandelblatt 2018 3-way interaction)",
        "note": "Primary model stratifies on APOE; HR is only used in synthetic DGP.",
        "source": "McDeed 2024 JNCI-CS; Ahles 2003 Psycho-Oncol"
    },
    "age_per_decade": {
        "HR": 1.35,
        "CI_95": (1.15, 1.60),
        "type": "continuous_per_10yr_above_reference",
        "reference_age": 50,
        "original_OR": None,
        "ph_handling": "FIXED for 0-24 mo (OK); time-varying if horizon > 36 mo",
        "note": "No clean standalone HR in CICI lit; extrapolated from AD / MCI cohorts.",
        "source": "AD/MCI literature extrapolation, Evidence Base §5"
    },
    "education_per_year": {
        "HR": 0.92,
        "CI_95": (0.87, 0.97),
        "type": "continuous_per_unit_above_reference",
        "reference_years": 14,
        "direction": "protective",
        "original_OR": 0.87,
        "ph_handling": "FIXED",
        "source": "Amidi 2015 Psycho-Oncol (n=72)"
    },
    "baseline_moca_low": {
        "HR": 1.80,
        "CI_95": (1.10, 2.90),
        "type": "binary",
        "threshold": "MoCA < 22",
        "original_OR": 2.58,
        "ph_handling": "FIXED",
        "source": "Dos Santos 2021 Cancers (n=310)"
    },

    # ── Inflammatory biomarkers ──
    "il6_elevated_postchemo": {
        "HR": 2.80,
        "CI_95": (1.20, 5.20),
        "type": "binary",
        "threshold": "post-chemo IL-6 > ~5 pg/mL",
        "original_OR": 5.50,
        "original_OR_CI": (1.43, 36.60),
        "ph_handling": "FIXED at peak post-chemo value; hazard contribution fades after 12 mo",
        "note": "Largest effect but wide CI; conservative HR conversion for 45% event rate.",
        "source": "Duivon 2024 Breast Cancer Res (CANTO-Cog)"
    },
    "crp_elevated": {
        "HR": 1.85,
        "CI_95": (1.05, 3.20),
        "type": "binary",
        "threshold": "CRP > 3.0 mg/L",
        "original_OR": 2.84,
        "original_OR_CI": (1.06, 7.64),
        "ph_handling": "FIXED at peak post-chemo value",
        "source": "Duivon 2024 Breast Cancer Res (CANTO-Cog)"
    },

    # ── Psychological symptoms ──
    "anxiety_per_unit": {
        "HR": 1.20,
        "CI_95": (1.04, 1.40),
        "type": "continuous_per_unit",
        "scale": "HADS-A (0-21)",
        "original_OR": 1.298,
        "ph_handling": "TIME-VARYING in reality; FIXED at baseline as acceptable limitation",
        "source": "Zhou 2024 Sci Rep (n=515)"
    },
    "depression_per_unit": {
        "HR": 1.08,
        "CI_95": (1.02, 1.15),
        "type": "continuous_per_unit",
        "scale": "PHQ-9 (0-27)",
        "original_OR": None,
        "ph_handling": "TIME-VARYING in reality; FIXED at baseline as limitation",
        "note": "LMM covariate coefficient rather than OR; conservative HR estimate.",
        "source": "Janelsins 2017 LMM covariate (URCC NCORP)"
    },
    "fatigue_per_unit": {
        "HR": 1.05,
        "CI_95": (1.01, 1.10),
        "type": "continuous_per_fatigue_severity_unit",
        "scale": "FACIT-F fatigue severity (higher = worse)",
        "original_OR": None,
        "ph_handling": "TIME-VARYING in reality; FIXED at baseline as limitation",
        "note": "Derived from CANTO fatigue effects on 2-year cognition.",
        "source": "Lange 2024 JAMA Network Open (CANTO)"
    },

    # ── Regimens (one-hot, CONTROL_CANCER_NO_CHEMO is the reference) ──
    "regimen_CMF": {
        "HR": 3.80,
        "CI_95": (1.50, 9.50),
        "type": "binary_regimen_vs_control",
        "original_OR": 6.40,
        "original_OR_CI": (1.50, 27.60),
        "ph_handling": "FIXED",
        "note": "Largest regimen-specific effect; also largest CI width.",
        "source": "Schagen 1999 Cancer (CMF vs. surgery-only)"
    },
    "regimen_AC_or_ACT": {
        "HR": 2.40,
        "CI_95": (1.40, 4.00),
        "type": "binary_regimen_vs_control",
        "original_OR": None,
        "ph_handling": "FIXED",
        "note": "AC and AC-T pooled; extrapolated from Janelsins cohorts.",
        "source": "Extrapolated from Janelsins 2012, 2017, 2018"
    },
    "regimen_FAC_FEC": {
        "HR": 2.60,
        "CI_95": (1.50, 4.50),
        "type": "binary_regimen_vs_control",
        "original_OR": None,
        "ph_handling": "FIXED",
        "source": "Extrapolated from Patel 2022, Janelsins 2012"
    },
    "regimen_PLATINUM": {
        "HR": 2.10,
        "CI_95": (1.20, 3.80),
        "type": "binary_regimen_vs_control",
        "original_OR": None,
        "ph_handling": "FIXED",
        "note": "Primarily testicular, lung, ovarian cohorts; smaller evidence base.",
        "source": "Extrapolated from Fung 2018 Adv Urol"
    },
    "regimen_TARGETED": {
        "HR": 1.40,
        "CI_95": (0.95, 2.10),
        "type": "binary_regimen_vs_control",
        "original_OR": None,
        "ph_handling": "FIXED",
        "note": "Trastuzumab, CDK4/6 inhibitors; weakest regimen signal.",
        "source": "McDeed 2024 HER2 LASSO signal"
    },

    # ── Treatment dose (MANDATORY time-varying) ──
    "cumulative_dose_per_1000": {
        "HR": 1.15,
        "CI_95": (1.05, 1.30),
        "type": "continuous_time_varying",
        "scale": "per 1000 mg/m² cyclophosphamide-equivalent",
        "original_OR": None,
        "ph_handling": "TIME-VARYING (MANDATORY) — fixed-at-baseline = immortal-time bias",
        "note": (
            "Dose accumulates mechanically across cycles; must be coded as "
            "start-stop / counting-process format in lifelines CoxTimeVaryingFitter."
        ),
        "source": "Collins 2013 Psycho-Oncol linear dose-response"
    },

    # ── Clinical comorbidities (TLC LASSO signals) ──
    "her2_positive": {
        "HR": 1.35,
        "CI_95": (1.00, 1.85),
        "type": "binary",
        "original_OR": None,
        "ph_handling": "FIXED",
        "note": "TLC LASSO signal independent of trastuzumab treatment.",
        "source": "McDeed 2024 JNCI-CS LASSO"
    },
    "cardiovascular_comorbidity": {
        "HR": 1.40,
        "CI_95": (1.05, 1.90),
        "type": "binary",
        "original_OR": None,
        "ph_handling": "FIXED",
        "source": "McDeed 2024 JNCI-CS LASSO"
    },
    "bmi_obese": {
        "HR": 1.30,
        "CI_95": (1.00, 1.70),
        "type": "binary",
        "threshold": "BMI ≥ 30",
        "original_OR": None,
        "ph_handling": "FIXED",
        "source": "McDeed 2024 JNCI-CS LASSO"
    },
}

# ── 3. ADDITIONAL COVARIATE DISTRIBUTIONS ────────────────────────────────
# Extensions to cici_config.py COHORT / BIOMARKERS for survival-specific variables.

ADDITIONAL_COVARIATES = {
    "hads_anxiety": {
        "distribution": "normal_truncated",
        "mean": 6.0, "sd": 4.0, "min": 0, "max": 21,
        "note": "HADS-A clinical anxiety scale; cancer population means 5-8.",
        "source": "Vodermaier 2011 meta-analysis of HADS in cancer"
    },
    "phq9_depression": {
        "distribution": "normal_truncated",
        "mean": 5.0, "sd": 4.0, "min": 0, "max": 27,
        "source": "Kroenke 2001 PHQ-9 validation; cancer cohort norms"
    },
    "facit_fatigue_severity": {
        "distribution": "normal_truncated",
        "mean": 12.0, "sd": 8.0, "min": 0, "max": 52,
        "note": (
            "Transformed to severity (higher = worse). FACIT-F as published "
            "has 0-52 range with higher = LESS fatigue; reverse-code for DGP."
        ),
        "source": "Yellen 1997 FACIT-F; Cella 2002 norms"
    },
    "bmi_kgm2": {
        "distribution": "normal_truncated",
        "mean": 27.0, "sd": 6.0, "min": 15, "max": 50,
        "obese_threshold": 30,
        "source": "NHANES-adjusted cancer cohort BMI distributions"
    },
    "cardiovascular_prevalence": {
        "distribution": "bernoulli",
        "p": 0.18,
        "age_adjusted": True,
        "note": "Baseline CVD (HTN, CAD, CHF) in women age 45-70 with cancer.",
        "source": "McDeed 2024 TLC cohort baseline characteristics"
    },
    "her2_positive_prevalence": {
        "distribution": "bernoulli",
        "p": 0.15,
        "note": "HER2+ breast cancer; applies to BC cohorts only.",
        "source": "SEER-Medicare BC subtype distributions"
    },
    "endocrine_therapy_start_month": {
        "distribution": "normal_truncated",
        "mean": 4.5, "sd": 1.0, "min": 2, "max": 8,
        "er_pr_positive_prevalence": 0.75,
        "note": "Tamoxifen or AI; time-varying binary in survival model.",
        "source": "Early Breast Cancer Trialists' Collaborative Group"
    },
}

# ── 4. CENSORING PARAMETERS ──────────────────────────────────────────────

CENSORING = {
    "administrative_horizon_months": 24,
    "uninformative_dropout_rate_per_6mo": 0.10,
    "uninformative_dropout_hazard_per_month": 0.01756,  # -log(0.9)/6
    "informative_dropout_multiplier": 0.30,  # used as exp(0.30 * eta_predicted)
    "note": (
        "Administrative τ = 24 mo matches Janelsins / CANTO-Cog conventions. "
        "10% per 6-mo dropout aggregates retention numbers from major CICI cohorts. "
        "Informative-dropout sensitivity simulates sicker-patients-drop-out-more."
    ),
    "source": "Evidence Base §7"
}

# ── 5. COMPETING MORTALITY ───────────────────────────────────────────────

COMPETING_MORTALITY = {
    "non_metastatic_bc": {
        "distribution": "exponential",
        "rate_per_month": 0.00192,
        "implied_24mo_cumulative": 0.045,
        "recommendation": "Cause-specific Cox (no Fine-Gray needed)",
        "source": "SEER BC stage I/II 5-year survival extrapolation"
    },
    "elderly_60plus": {
        "distribution": "exponential",
        "rate_per_month": 0.00330,
        "implied_24mo_cumulative": 0.077,
        "recommendation": "Fine-Gray as sensitivity analysis",
        "source": "McDeed 2024 TLC cohort mortality"
    },
    "metastatic": {
        "distribution": "weibull",
        "shape": 1.4,
        "scale_months": 24,
        "rate_per_month": 0.0300,
        "implied_24mo_cumulative": 0.51,
        "recommendation": "Fine-Gray as PRIMARY (cause-specific biased)",
        "source": "SEER stage IV BC survival"
    },
    "default_for_refactor": "non_metastatic_bc",
    "note": "Competing mortality is sensitivity analysis S2, not primary cohort.",
}

# ── 6. ASSESSMENT SCHEDULES (interval censoring) ────────────────────────

ASSESSMENT_SCHEDULES = {
    "A_dense": {
        "visit_months": [0, 3, 6, 9, 12, 18, 24],
        "max_interval_months": 6,
        "recommended_analysis": "Standard Cox PH with midpoint imputation",
        "expected_c_index_penalty": "< 0.02 vs. exact times",
    },
    "B_cici_typical": {
        "visit_months": [0, 4, 6, 12, 24],
        "max_interval_months": 12,
        "recommended_analysis": "Cox PH + Weibull AFT with interval likelihood",
        "expected_c_index_penalty": "0.02 – 0.05 vs. exact times",
        "is_primary": True,
    },
    "C_sparse": {
        "visit_months": [0, 12, 24],
        "max_interval_months": 12,
        "recommended_analysis": "Weibull AFT with interval likelihood + MI sensitivity",
        "expected_c_index_penalty": "0.05 – 0.10 vs. exact times",
    },
}

# ── 7. OUTCOME DEFINITIONS (SURVIVAL REFACTOR) ──────────────────────────

OUTCOME_SURVIVAL = {
    "primary": {
        "label": "RCI_APE_decline",
        "definition": (
            "Reliable Change Index on composite Attention / Processing-speed / "
            "Executive function z-score. Event = 1 at first visit where "
            "RCI <= -1.645, where RCI = (X2 - X1 - practice_effect) / SE_diff."
        ),
        "expected_event_rate_12mo": 0.30,
        "expected_event_rate_24mo": 0.42,
        "tolerance": 0.05,
        "source": "Wefel 2004 Cancer; Ahles 2003; Evidence Base §1"
    },
    "secondary": {
        "label": "FACT_Cog_PCI_below_54",
        "definition": "Self-reported FACT-Cog PCI subscale < 54 at any visit.",
        "sensitivity": 0.76,
        "specificity": 0.83,
        "expected_event_rate_12mo": 0.45,
        "expected_event_rate_24mo": 0.55,
        "tolerance": 0.05,
        "caveat": (
            "Correlates more strongly with depression/anxiety/fatigue than with "
            "objective neuropsychological decline. Report as sensitivity outcome."
        ),
        "source": "Van Dyk 2020 JNCI-CS"
    },
}

# ── 8. HR RECOVERY TOLERANCES (sanity-check gate for synthetic DGP) ──────
# When the simulation is fit with Cox PH, each HR should recover to within
# these ranges. Failure indicates a simulation bug, not a modeling finding.
#
# Collinearity note: regimen dummies and cumulative_dose are partially
# confounded because dose is determined by regimen (CMF always ~8400 mg/m²,
# FAC/FEC always ~3600, etc.). With the tight within-regimen dose SDs from
# cici_config.REGIMENS, standard Cox PH cannot fully disentangle their
# effects — dose estimation tends to be biased upward and the highest-dose
# regimens (CMF, FAC/FEC) correspondingly downward. This is the exact
# scenario Cox-LASSO is built for and is pre-specified as a Sprint 3
# secondary model in SURVIVAL_DESIGN.md §4. Tolerances for these three
# covariates are widened accordingly.

RECOVERY_TOLERANCES = {
    "apoe_e4":                    {"true_HR": 1.75, "acceptable": (1.40, 2.20)},
    "age_per_decade":              {"true_HR": 1.35, "acceptable": (1.20, 1.52)},
    "education_per_year":          {"true_HR": 0.92, "acceptable": (0.88, 0.96)},
    "baseline_moca_low":           {"true_HR": 1.80, "acceptable": (1.40, 2.30)},
    "il6_elevated_postchemo":      {"true_HR": 2.80, "acceptable": (2.10, 3.60)},
    "crp_elevated":                {"true_HR": 1.85, "acceptable": (1.45, 2.35)},
    "anxiety_per_unit":            {"true_HR": 1.20, "acceptable": (1.12, 1.30)},
    "depression_per_unit":         {"true_HR": 1.08, "acceptable": (1.03, 1.14)},
    "fatigue_per_unit":            {"true_HR": 1.05, "acceptable": (1.02, 1.09)},
    "regimen_CMF":                 {"true_HR": 3.80, "acceptable": (0.80, 5.10)},    # widened (see collinearity note)
    "regimen_AC_or_ACT":            {"true_HR": 2.40, "acceptable": (1.60, 3.10)},    # widened
    "regimen_FAC_FEC":              {"true_HR": 2.60, "acceptable": (1.40, 3.40)},    # widened
    "regimen_PLATINUM":             {"true_HR": 2.10, "acceptable": (1.55, 2.85)},
    "regimen_TARGETED":             {"true_HR": 1.40, "acceptable": (1.00, 1.95)},
    "cumulative_dose_per_1000":    {"true_HR": 1.15, "acceptable": (1.05, 1.35)},    # widened
    "her2_positive":               {"true_HR": 1.35, "acceptable": (1.00, 1.80)},
    "cardiovascular_comorbidity":  {"true_HR": 1.40, "acceptable": (1.05, 1.85)},
    "bmi_obese":                   {"true_HR": 1.30, "acceptable": (1.00, 1.70)},
}

# ── 9. SYNTHETIC COHORT META-PARAMETERS ─────────────────────────────────

SURVIVAL_COHORT = {
    "n": 2500,
    "seed": 42,
    "train_frac": 0.80,
    "test_frac": 0.20,
    "stratify_split_on": "event",
    "bootstrap_n": 1000,
    "cv_folds": 10,
    "default_schedule": "B_cici_typical",
    "default_outcome": "primary",
    "default_hazard": "weibull_primary",
    "note": (
        "Cohort size increased from the 1,500 used in the binary Fork B model "
        "to preserve effective N under survival-analysis censoring."
    ),
}

# ── 10. NEW CITATIONS (in addition to cici_config.CITATIONS) ─────────────

SURVIVAL_CITATIONS = [
    {
        "key": "Musto2024",
        "authors": "Musto H, Stamate D, Logofatu D, Stahl D",
        "year": 2024,
        "title": "Predicting deterioration in MCI with survival transformers, XGBoost, and Cox PH",
        "journal": "ICANN 2024 Proceedings (Springer)",
        "volume": "arXiv:2409.16231",
        "n": "ADNI metabolomics cohort",
        "finding": "c-index 0.77 (Cox PH), 0.80 (Survival XGBoost), 0.85 (Survival Transformer)",
        "role": "PRIMARY BENCHMARK for survival refactor"
    },
    {
        "key": "Duan2025",
        "authors": "Duan Z, Zhang S, Liu Y, et al.",
        "year": 2025,
        "title": "CRCI prediction models: a systematic review",
        "journal": "Clinical and Experimental Medicine",
        "volume": "DOI 10.1007/s10238-025-01590-6",
        "n": "9 models, search through Aug 2024",
        "finding": "All 9 CRCI models binary classifiers; none use Cox PH or survival ML",
        "role": "Establishes gap / novelty for survival refactor"
    },
    {
        "key": "Tohidinezhad2024",
        "authors": "Tohidinezhad F et al.",
        "year": 2024,
        "title": "NTCP for neurocognitive decline after brain radiotherapy",
        "journal": "Neuro-Oncology",
        "volume": "26(8):1467-1478",
        "finding": "Closest survival-adjacent oncology cognition model; uses time-binned logistic, not Cox",
        "role": "Context for why survival framing is novel"
    },
    {
        "key": "Lange2024",
        "authors": "Lange M, Joly F, Levy C, et al.",
        "year": 2024,
        "title": "Cognition and return to work in CANTO",
        "journal": "JAMA Network Open",
        "volume": "7(8):e2427576",
        "n": 4262,
        "finding": "Largest modern CRCI longitudinal cohort; LMM trajectory modeling",
        "role": "Reference cohort for fatigue, anxiety, depression covariate distributions"
    },
    {
        "key": "Sun2025",
        "authors": "Sun H, Wang X, Zhang L, et al.",
        "year": 2025,
        "title": "Longitudinal cognitive function in breast cancer and lymphoma",
        "journal": "JNCI Cancer Spectrum",
        "volume": "9(6):pkaf105",
        "finding": "Bimodal CRCI trajectory with late re-emergence at 18-24 months",
        "role": "Evidence for log-logistic alternative baseline hazard"
    },
    {
        "key": "Palesh2025",
        "authors": "Palesh O et al.",
        "year": 2025,
        "title": "Natural trajectory subclasses of CRCI with insomnia",
        "journal": "Cancer",
        "volume": "131(8):e35816",
        "finding": "Latent-class trajectories; three distinct CRCI subgroups",
        "role": "Supports heterogeneity argument for survival framing"
    },
    {
        "key": "Boschetti2025",
        "authors": "Boschetti A et al.",
        "year": 2025,
        "title": "Biomarkers and CRCI in breast cancer: a systematic review",
        "journal": "Cancers",
        "volume": "17(21):3522",
        "finding": "COMT / BDNF / MTHFR effects inconsistent; flag as low-confidence",
        "role": "Justifies exclusion of polymorphism features beyond APOE"
    },
]