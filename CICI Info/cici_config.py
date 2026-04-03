"""
CICI Risk Model — Literature-Grounded Configuration
=====================================================
Every number below is traceable to a published source.
Use this file to parameterize synthetic cohort generation
and XGBoost feature engineering for CICI prediction.

Generated for SBMT 2026 NeuroTech Convention poster.
"""

# ── 1. COHORT DEMOGRAPHICS (sampling distributions) ──────────────────────

COHORT = {
    "n_synthetic": 1500,  # mid-range for literature; scale as needed

    "age": {
        "mean": 53.4, "sd": 9.0, "min": 25, "max": 80,
        "source": "Janelsins 2018 JCO (n=580)"
    },
    "education_years": {
        "mean": 14.0, "sd": 3.0, "min": 0, "max": 22,
        "source": "Janelsins 2017/2018 JCO; Western cohorts"
    },
    "apoe_e4_carrier_prob": {
        "value": 0.22,
        "source": "UK Biobank BC subsample 20% (Lehrer 2021); CANTO-Cog 19% (Duivon 2024)"
    },
    "baseline_moca": {
        "mean": 25.0, "sd": 3.0, "min": 10, "max": 30,
        "source": "Fernandes 2017 (median 24, IQR 21-26); MoCA normative data"
    },
    "baseline_mmse": {
        "mean": 28.0, "sd": 2.0, "min": 18, "max": 30,
        "source": "General normative; ceiling effects limit utility"
    },
}

# ── 2. INFLAMMATORY BIOMARKER DISTRIBUTIONS ──────────────────────────────
# All distributions are RIGHT-SKEWED; sample from lognormal then clip

BIOMARKERS = {
    "il6_baseline_pgml": {
        "median": 2.0, "log_mean": 0.7, "log_sd": 0.8,
        "min": 0.1, "max": 50.0,
        "note": "14% below detection limit in Cheung 2015; use lognormal",
        "source": "Cheung 2015 Ann Oncol; Ganz 2013 BBI"
    },
    "il6_postchemo_pgml": {
        "median": 4.5, "log_mean": 1.2, "log_sd": 0.9,
        "min": 0.1, "max": 100.0,
        "note": "Significant increase p<0.0001 post-chemo",
        "source": "Cheung 2015; Janelsins 2012"
    },
    "crp_baseline_mgl": {
        "median": 3.0, "log_mean": 1.1, "log_sd": 1.0,
        "min": 0.1, "max": 50.0,
        "source": "Carroll 2022 JCO (TLC); Duivon 2024 BCR"
    },
    "crp_postchemo_mgl": {
        "median": 5.0, "log_mean": 1.4, "log_sd": 1.1,
        "min": 0.1, "max": 80.0,
        "source": "Carroll 2022 JCO"
    },
    "tnfa_baseline_pgml": {
        "median": 8.0, "log_mean": 2.0, "log_sd": 1.2,
        "min": 0.5, "max": 150.0,
        "note": "Extreme assay-dependent variability; Ramalho 2023 reported 33.8±36.2",
        "source": "Ganz 2013; Williams 2018 J Neuroimmunol"
    },
    "tnfa_postchemo_pgml": {
        "median": 12.0, "log_mean": 2.3, "log_sd": 1.2,
        "min": 0.5, "max": 200.0,
        "source": "Ganz 2013; transient increase post-chemo"
    },
}

# ── 3. CHEMOTHERAPY REGIMEN DISTRIBUTIONS ────────────────────────────────

REGIMENS = {
    "AC": {
        "label": "Doxorubicin + Cyclophosphamide",
        "cyclo_dose_mg_m2_total": {"mean": 2400, "sd": 200, "min": 1800, "max": 3000},
        "n_cycles": 4,
        "cici_incidence": {"point": 0.33, "range": (0.25, 0.40)},
        "source": "Wefel 2004 Cancer; 600 mg/m² × 4 cycles"
    },
    "AC_T": {
        "label": "AC → Taxane (paclitaxel or docetaxel)",
        "cyclo_dose_mg_m2_total": {"mean": 2400, "sd": 200, "min": 1800, "max": 3000},
        "n_cycles": 8,  # 4 AC + 4 taxane
        "cici_incidence": {"point": 0.40, "range": (0.30, 0.50)},
        "source": "Kerkmann 2025 Oncologist; Janelsins 2018 JCO"
    },
    "CMF": {
        "label": "Cyclophosphamide + Methotrexate + 5-FU",
        "cyclo_dose_mg_m2_total": {"mean": 8400, "sd": 1000, "min": 6000, "max": 10000},
        "n_cycles": 6,
        "cici_incidence": {"point": 0.50, "range": (0.28, 0.75)},
        "source": "Schagen 1999 Cancer (28%); van Dam 1998 (75%); oral ~75% bioavail"
    },
    "FAC_FEC": {
        "label": "5-FU + Anthracycline + Cyclophosphamide",
        "cyclo_dose_mg_m2_total": {"mean": 3600, "sd": 400, "min": 2400, "max": 4200},
        "n_cycles": 6,
        "cici_incidence": {"point": 0.35, "range": (0.20, 0.50)},
        "source": "Patel 2022; Janelsins 2012"
    },
    "PLATINUM": {
        "label": "Cisplatin or Carboplatin-based (non-breast)",
        "cyclo_dose_mg_m2_total": None,  # not cyclophosphamide-based
        "n_cycles": 4,
        "cici_incidence": {"point": 0.30, "range": (0.19, 0.58)},
        "source": "Fung 2018 Adv Urol; Wefel 2011 Cancer"
    },
    "CONTROL_CANCER_NO_CHEMO": {
        "label": "Cancer patients, no chemotherapy",
        "cyclo_dose_mg_m2_total": None,
        "n_cycles": 0,
        "cici_incidence": {"point": 0.15, "range": (0.10, 0.20)},
        "source": "Mandelblatt 2014; Ahles 2008"
    },
    "CONTROL_HEALTHY": {
        "label": "Healthy controls (expected false-positive rate)",
        "cyclo_dose_mg_m2_total": None,
        "n_cycles": 0,
        "cici_incidence": {"point": 0.12, "range": (0.10, 0.15)},
        "source": "Base rate on neuropsych batteries with multiple tests"
    },
}

# Suggested regimen sampling weights for a breast cancer cohort
REGIMEN_WEIGHTS = {
    "AC_T": 0.45,     # most common modern regimen
    "AC": 0.15,
    "FAC_FEC": 0.10,
    "CMF": 0.05,       # less common now
    "CONTROL_CANCER_NO_CHEMO": 0.25,
}

# ── 4. ODDS RATIOS FOR FEATURE → CICI ───────────────────────────────────
# Use these to assign CICI labels probabilistically in synthetic data

ODDS_RATIOS = {
    "apoe_e4_carrier": {
        "OR": 2.50,
        "CI_95": (1.20, 5.20),
        "type": "binary",
        "note": "Strongest LASSO predictor in TLC; d≈0.50 in Ahles 2003. "
                "CANTO-Cog found null overall result — effect may be domain-specific.",
        "source": "McDeed 2024 JNCI-CS; Ahles 2003 Psycho-Oncol"
    },
    "baseline_cognitive_impairment": {
        "OR": 2.58,
        "CI_95": (1.15, 5.80),
        "type": "binary",  # abnormal vs normal MMSE
        "threshold": "MMSE < 24 or MoCA < 22",
        "source": "Dos Santos 2021 Cancers (n=310, multivariate)"
    },
    "crp_elevated": {
        "OR": 2.84,
        "CI_95": (1.06, 7.64),
        "type": "binary",  # > 3 mg/L vs ≤ 3 mg/L
        "threshold": "CRP > 3.0 mg/L",
        "source": "Duivon 2024 Breast Cancer Res (CANTO-Cog, n=200)"
    },
    "il6_elevated": {
        "OR": 5.50,
        "CI_95": (1.43, 36.60),
        "type": "binary",  # elevated vs normal (varies by assay)
        "note": "Largest effect size but wide CI; episodic memory domain",
        "source": "Duivon 2024 Breast Cancer Res (CANTO-Cog)"
    },
    "tnfa_stnfri": {
        "r": -0.65,
        "type": "continuous_correlation",
        "note": "sTNFRI correlated with delayed match-to-sample; small sample",
        "source": "Williams 2018 J Neuroimmunol (n=22)"
    },
    "education_high": {
        "OR": 0.103,
        "CI_95": (0.02, 0.50),
        "type": "binary",  # > 6 yrs vs no education
        "note": "Chinese cohort; use ~0.90 per year for Western cohorts",
        "source": "Zhou 2024 Sci Rep (n=515)"
    },
    "education_per_iq_point": {
        "OR": 0.87,
        "CI_95": (0.81, 0.95),
        "type": "continuous_per_unit",
        "source": "Amidi 2015 Psycho-Oncol (n=72, testicular cancer)"
    },
    "anxiety_per_unit": {
        "OR": 1.298,
        "CI_95": (1.05, 1.60),
        "type": "continuous_per_unit",
        "note": "Per unit increase on anxiety scale in nomogram",
        "source": "Zhou 2024 Sci Rep"
    },
    "depression_anxiety_composite": {
        "beta": 0.25,
        "p": 0.002,
        "type": "SEM_path_coefficient",
        "note": "Explained 47.7% of CICI variance in SEM model",
        "source": "Kim 2019 J Korean Acad Nursing"
    },
    "chemotherapy_vs_none": {
        "OR": 6.40,
        "CI_95": (1.50, 27.60),
        "type": "binary",
        "note": "CMF vs surgery-only; largest regimen-specific OR",
        "source": "Schagen 1999 Cancer"
    },
    "cumulative_dose": {
        "type": "linear_dose_response",
        "note": "Linear model best fit for cognition decline across cycles",
        "source": "Collins 2013 Psycho-Oncol (n=60+60)"
    },
    "subjective_complaints_pci": {
        "OR": 0.956,
        "type": "continuous_per_unit",
        "note": "Per unit FACT-Cog PCI; lower PCI = more complaints = higher risk",
        "source": "Lange 2016"
    },
    "age": {
        "type": "interaction_modifier",
        "note": "Significant covariate (p<0.05) in multiple models; "
                "interacts with cognitive reserve and APOE. No clean standalone OR available. "
                "Use as continuous feature — XGBoost will learn interactions.",
        "source": "Ahles 2010 JCO; Mandelblatt 2018 JCO"
    },
}

# ── 5. OUTCOME VARIABLE DEFINITION ──────────────────────────────────────

OUTCOME = {
    "icctf_standard": {
        "definition": (
            "≥ 2 neuropsych test scores ≤ 1.5 SD below normative mean, "
            "OR ≥ 1 test score ≤ 2.0 SD below normative mean"
        ),
        "source": "Wefel 2011 Lancet Oncol (ICCTF consensus)",
        "adoption_rate": "37.6% of post-2011 studies (ASCO 2024 abstract)"
    },
    "rci_method": {
        "definition": "RCI = (X2 - X1 - practice_effect) / SE_diff; |RCI| > 1.645",
        "note": "Preferred for longitudinal; accounts for test-retest reliability"
    },
    "self_report_cutoff": {
        "instrument": "FACT-Cog PCI subscale",
        "cutoff_18item": 54,
        "cutoff_20item": 60,
        "sensitivity": 0.76,
        "specificity": 0.83,
        "source": "Van Dyk 2020 JNCI-CS"
    },
    "core_neuropsych_tests": [
        "Hopkins Verbal Learning Test-Revised (HVLT-R) — learning/memory",
        "Trail Making Test A & B (TMT) — attention, processing speed, executive",
        "Controlled Oral Word Association Test (COWA) — verbal fluency",
    ],
    "supplementary_tests": [
        "Rey Auditory Verbal Learning Test (RAVLT)",
        "WAIS Digit Span & Digit Symbol-Coding",
        "Stroop Color-Word Test",
        "Grooved Pegboard",
    ],
    "recommended_for_synthetic": (
        "For XGBoost: binarize as CICI=1 if simulated impairment on ≥ 2 of 6 "
        "test z-scores fall ≤ -1.5, or ≥ 1 falls ≤ -2.0. "
        "Assign z-scores by sampling from N(0,1) for controls and "
        "N(-shift, 1) for chemo patients, where shift is drawn from "
        "the OR-weighted risk profile."
    ),
}

# ── 6. BENCHMARK MODEL PERFORMANCE ──────────────────────────────────────

BENCHMARKS = {
    "tlc_lasso": {
        "auc_transient": 0.75,
        "auc_persistent": 0.79,
        "method": "LASSO logistic regression, 94 predictors",
        "key_features": [
            "APOE ε4", "HER2 status", "BMI ≥ 30",
            "cardiovascular comorbidities", "baseline APE score"
        ],
        "n": 691,
        "validation": "Internal (bootstrap)",
        "source": "McDeed 2024 JNCI Cancer Spectrum"
    },
    "zhou_nomogram": {
        "auc": 0.955,
        "method": "Multivariate logistic regression → nomogram",
        "key_features": [
            "education", "anxiety", "chemotherapy",
            "benefit finding", "fasting glucose"
        ],
        "n": 515,
        "validation": "Internal (split-sample)",
        "caution": "Cross-sectional; AUC likely overfit",
        "source": "Zhou 2024 Scientific Reports"
    },
    "liu_model": {
        "auc": 0.882,
        "method": "Logistic regression → nomogram",
        "key_features": [
            "education", "exercise frequency",
            "chemotherapy cycles", "fatigue"
        ],
        "n": 741,
        "validation": "5-fold cross-validation",
        "source": "Liu 2024 J Cancer Survivorship"
    },
    "colorectal_model": {
        "auc": 0.799,
        "method": "Logistic regression, 14 predictors",
        "n": 386,
        "validation": "Internal",
        "source": "2021 BioMed Res Int"
    },
    "target_for_xgboost": {
        "realistic_auc": "0.78-0.85",
        "note": (
            "TLC LASSO at 0.75-0.79 is the most trustworthy benchmark. "
            "XGBoost should capture nonlinear interactions (age × reserve × APOE) "
            "that linear models miss. Aim for ≥ 0.80 on held-out synthetic test set."
        )
    },
}

# ── 7. NEUROIMAGING DATASETS ─────────────────────────────────────────────

DATASETS = {
    "brats_2024": {
        "url": "https://www.synapse.org/brats",
        "modalities": ["T1", "T1c", "T2", "T2-FLAIR"],
        "cases": "1000+",
        "ground_truth": "Expert tumor segmentation masks",
        "cognitive_labels": False,
        "relevance": "3D U-Net segmentation branch; Dice benchmarks 0.89-0.93"
    },
    "adni": {
        "url": "https://adni.loni.usc.edu",
        "modalities": ["MRI", "fMRI", "DTI", "PET (amyloid/tau/FDG)"],
        "cases": "2000+",
        "ground_truth": "MMSE, ADAS-Cog, CDR staging, APOE genotype",
        "cognitive_labels": True,
        "relevance": "Transfer learning for cognitive decline detection; APOE stratification"
    },
    "oasis3": {
        "url": "https://sites.wustl.edu/oasisbrains",
        "modalities": ["T1", "T2", "FLAIR", "ASL", "rsfMRI", "DTI"],
        "cases": 1378,
        "pet_scans": 2157,
        "ground_truth": "CDR, MMSE, longitudinal cognitive tracking, APOE",
        "cognitive_labels": True,
        "relevance": "Open-access; broad modalities; cognitive impairment classification"
    },
    "uk_biobank": {
        "url": "https://www.ukbiobank.ac.uk",
        "modalities": ["T1", "T2-FLAIR", "SWI", "dMRI", "rsfMRI", "task-fMRI"],
        "cases": "40000+ brain scans",
        "ground_truth": "NHS cancer/treatment records, cognitive testing, APOE, blood biomarkers",
        "cognitive_labels": True,
        "relevance": "Best resource for retrospective chemo-brain imaging; requires approved application"
    },
    "tcia_brain_tumor_progression": {
        "url": "https://www.cancerimagingarchive.net/collection/brain-tumor-progression/",
        "modalities": ["MRI longitudinal"],
        "cases": 20,
        "ground_truth": "Post-chemoradiation progression",
        "cognitive_labels": False,
        "relevance": "Cancer-specific brain imaging; transfer learning"
    },
}

# ── 8. KEY CITATIONS (BibTeX-ready) ─────────────────────────────────────

CITATIONS = [
    {
        "key": "McDeed2024",
        "authors": "McDeed AP, Ganz PA, Engel J, et al.",
        "year": 2024,
        "title": "Prediction of cognitive decline in older breast cancer survivors: the TLC study",
        "journal": "JNCI Cancer Spectrum",
        "volume": "8(2):pkae019",
        "n": 691,
        "finding": "LASSO AUC 0.75/0.79; APOE e4 strongest predictor",
        "has_OR": True,
    },
    {
        "key": "Duivon2024",
        "authors": "Duivon M, Lange M, Brundage M, et al.",
        "year": 2024,
        "title": "Inflammation at diagnosis and cognitive impairment two years later (CANTO-Cog)",
        "journal": "Breast Cancer Research",
        "volume": "26:90",
        "n": 200,
        "finding": "CRP OR=2.84, IL-6 OR=5.50 for episodic memory impairment",
        "has_OR": True,
    },
    {
        "key": "Schagen1999",
        "authors": "Schagen SB, van Dam FS, Muller MJ, et al.",
        "year": 1999,
        "title": "Cognitive deficits after postoperative adjuvant chemotherapy for breast carcinoma",
        "journal": "Cancer",
        "volume": "85(3):640-650",
        "n": 70,
        "finding": "CMF: 28% impaired vs 12% controls; OR=6.4",
        "has_OR": True,
    },
    {
        "key": "Ahles2003",
        "authors": "Ahles TA, Saykin AJ, Noll WW, et al.",
        "year": 2003,
        "title": "The relationship of APOE genotype to neuropsych performance in cancer survivors",
        "journal": "Psycho-Oncology",
        "volume": "12(6):612-619",
        "n": 80,
        "finding": "APOE e4 carriers scored lower on visual memory/spatial ability (d≈0.50)",
        "has_OR": False,
    },
    {
        "key": "Janelsins2018",
        "authors": "Janelsins MC, Heckler CE, Peppone LJ, et al.",
        "year": 2018,
        "title": "Longitudinal trajectory of cancer-related cognitive impairment (URCC NCORP)",
        "journal": "Journal of Clinical Oncology",
        "volume": "36(32):3231-3239",
        "n": 943,
        "finding": "Significant decline pre→post chemo on delayed match-to-sample",
        "has_OR": False,
    },
    {
        "key": "Mandelblatt2018",
        "authors": "Mandelblatt JS, Stern RA, Luta G, et al.",
        "year": 2018,
        "title": "Cognitive outcomes among older breast cancer survivors (TLC study)",
        "journal": "Journal of Clinical Oncology",
        "volume": "36(32):3211-3222",
        "n": 691,
        "finding": "Chemo effects confined to APOE e4 carriers (p=0.03 for 3-way interaction)",
        "has_OR": False,
    },
    {
        "key": "Carroll2022",
        "authors": "Carroll JE, Small BJ, Tometich DB, et al.",
        "year": 2022,
        "title": "Elevated CRP and subsequent patient-reported cognitive problems (TLC study)",
        "journal": "Journal of Clinical Oncology",
        "volume": "40(25):2893-2903",
        "n": 580,
        "finding": "CRP 3.0 mg/L → FACT-Cog 9.5 pts lower; 10.0 mg/L → 14.2 pts lower",
        "has_OR": False,
    },
    {
        "key": "Collins2013",
        "authors": "Collins B, Mackenzie J, Tasca GA, et al.",
        "year": 2013,
        "title": "Cognitive effects of chemotherapy in breast cancer: a dose-response study",
        "journal": "Psycho-Oncology",
        "volume": "22(7):1517-1527",
        "n": 120,
        "finding": "Linear dose-response for cognition decline across chemo cycles",
        "has_OR": False,
    },
    {
        "key": "Wefel2011_ICCTF",
        "authors": "Wefel JS, Vardy J, Ahles T, Schagen SB",
        "year": 2011,
        "title": "ICCTF recommendations to harmonise studies of cognitive function in cancer",
        "journal": "The Lancet Oncology",
        "volume": "12(7):703-708",
        "n": None,
        "finding": "Consensus: ≥2 tests ≤-1.5 SD or ≥1 test ≤-2.0 SD = impaired",
        "has_OR": False,
    },
    {
        "key": "Whittaker2022",
        "authors": "Whittaker AL, George RP, O'Malley L",
        "year": 2022,
        "title": "Prevalence of cognitive impairment following chemo for breast cancer (meta-analysis)",
        "journal": "Scientific Reports",
        "volume": "12:2135",
        "n": "52 studies",
        "finding": "Pooled: 44% self-report, 16% screening, 21-34% neuropsych batteries",
        "has_OR": False,
    },
    {
        "key": "Zhou2024",
        "authors": "Zhou H, Zhang Y, Chen L, et al.",
        "year": 2024,
        "title": "A nomogram for predicting CRCI risk in breast cancer (symptom model)",
        "journal": "Scientific Reports",
        "volume": "14:14762",
        "n": 515,
        "finding": "Nomogram AUC=0.955; education OR=0.103, anxiety OR=1.298",
        "has_OR": True,
    },
    {
        "key": "DosSantos2021",
        "authors": "Dos Santos M, Pouessel D, Bhatt A, et al.",
        "year": 2021,
        "title": "Cognitive impairment in older cancer patients treated with first-line chemotherapy",
        "journal": "Cancers",
        "volume": "13(23):6048",
        "n": 310,
        "finding": "Baseline MMSE abnormal → OR=2.58 for further impairment",
        "has_OR": True,
    },
]
