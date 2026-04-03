# Chemo brain: a comprehensive evidence base for CICI risk prediction

**Chemotherapy-induced cognitive impairment (CICI) affects roughly one in three cancer patients by objective testing**, yet no externally validated clinical prediction tool exists to identify high-risk individuals before treatment. This report synthesizes quantitative data from prospective cohorts, meta-analyses, and machine learning studies to provide a complete evidence base for building a tabular XGBoost risk prediction model. The strongest published predictors include APOE ε4 carrier status, baseline cognitive reserve, inflammatory markers (CRP, IL-6), and anxiety—with odds ratios ranging from 1.3 to 5.5. Existing proof-of-concept models achieve AUCs of 0.75–0.88 using logistic regression and LASSO, establishing a clear benchmark for more sophisticated approaches. The absence of a validated CICI risk calculator represents both a significant clinical gap and a meaningful research opportunity.

---

## 1. Incidence rates vary dramatically by regimen and measurement method

The reported prevalence of CICI ranges from **14% to 85%** across studies, driven largely by how impairment is measured. A meta-analysis of 52 studies found pooled prevalence of **44% by self-report**, **16% by short cognitive screens**, and **21–34% by neuropsychological test batteries** (Whittaker et al., 2022, *Scientific Reports*). A more recent meta-analysis of 11 breast cancer studies reported pooled incidence of **39.9%** (95% CI: 26.3–55.2%) (Grégoire et al., 2025, *Scientific Reports*). Up to **75%** of patients experience some cognitive disruption during active treatment, with roughly **35%** showing persistent symptoms months to years afterward (Janelsins et al., 2014, *International Review of Psychiatry*).

**Cyclophosphamide-based regimens (CMF, AC)** carry some of the strongest evidence. Schagen et al. (1999, *Cancer*) reported **28% cognitive impairment** in CMF-treated patients versus 12% in surgery-only controls, yielding **OR = 6.4** (95% CI: 1.5–27.6, p = 0.013). Van Dam et al. (1998) found **75%** of CMF patients scored ≥2 SD below norms on at least one test 3–18 months post-treatment. Effects persist remarkably long: CMF-treated survivors performed worse on learning, memory, processing speed, and psychomotor functioning **21 years** after treatment (Koppelmans et al., 2012, *Journal of Clinical Oncology*; n = 196). For AC regimens, Wefel et al. (2004, *Cancer*) found approximately **33%** showed cognitive decline post-treatment in the first prospective longitudinal study of CICI.

**Anthracycline-based regimens** show variable results. FAC (5-FU/doxorubicin/cyclophosphamide) produced a **6.05% MMSE decline** over three cycles versus 2.68% for CMF and 3.80% for CEF (Patel et al., 2022, *PMC*), consistent with anthracyclines' greater neuroinflammatory potential—Janelsins et al. (2012, *Supportive Care in Cancer*) demonstrated that anthracycline-containing regimens produced significantly higher IL-6, MCP-1, and IL-8 levels than non-anthracycline regimens. Estimated incidence ranges from **20–50%** depending on assessment timing. Interestingly, FEC (epirubicin-based) showed **no elevated risk** versus controls at 2 years in Schagen et al. (2006, *Annals of Oncology*).

**AC-T (anthracycline → taxane sequential)**, the most common breast cancer regimen, has limited isolated data because most studies combine it with other regimens. The CICARO prospective cohort found approximately **33% measurable impairment** 2–4 weeks post-chemotherapy in patients receiving predominantly AC-based ± taxane regimens, with recovery by 2–3 years (Kerkmann et al., 2025, *The Oncologist*). Janelsins et al. (2018, *Journal of Clinical Oncology*; n = 580 patients, 363 controls) demonstrated significant decline on delayed match-to-sample testing from pre- to post-chemotherapy in AC ± taxane recipients. Estimated range: **30–50% by objective testing; up to 70% by self-report**.

**Platinum-based regimens** in non-breast cancers show notable impairment. In testicular cancer treated with BEP (bleomycin/etoposide/cisplatin), **46–58%** had cognitive impairment even before chemotherapy—higher than expected (Wefel et al., 2011, *Cancer*; Buskbjerg et al., 2021, *Cancer Medicine*). Post-treatment, **19%** still had worsened cognitive function at 2 years in a prospective EORTC trial (n = 666; Fung et al., 2018, *Advances in Urology*). Colorectal cancer patients on oxaliplatin-based regimens show **13–57%** impairment rates.

**Control rates** provide essential baselines. Non-chemotherapy breast cancer patients show approximately **10–20%** cognitive impairment, while **20–30%** of all cancer patients show some impairment before any treatment begins (Ahles et al., 2008; Mandelblatt et al., 2014; Lange et al., 2020). Healthy controls typically show **10–15%** impairment rates on neuropsychological batteries (expected false-positive base rate with multiple tests).

| Regimen | Incidence (objective) | Incidence (self-report) | Key source |
|---|---|---|---|
| CMF | 28–75% | — | Schagen 1999; van Dam 1998 |
| AC | ~33% | — | Wefel et al. 2004 |
| AC-T | 30–50% | Up to 70% | Kerkmann 2025; Janelsins 2018 |
| FAC/anthracycline | 20–50% | — | Patel 2022; Janelsins 2012 |
| Platinum (BEP) | 19–32% persistent | — | Fung 2018; Schagen 2008 |
| Non-chemo cancer | 10–20% | — | Mandelblatt 2014 |
| Healthy controls | 10–15% | — | Expected base rate |

---

## 2. Odds ratios and effect sizes for candidate XGBoost features

Published effect estimates for CICI risk factors span genetic, inflammatory, cognitive, and psychological domains. The table below synthesizes the best available evidence for each candidate predictor, noting whether estimates come from univariate or multivariate analyses.

**APOE ε4 carrier status** is among the most consistently identified genetic risk factors. The Thinking and Living with Cancer (TLC) study's LASSO model found APOE ε4 was the **single strongest predictor** of both transient and persistent cognitive decline—survivors with APOE ε4+, cardiovascular comorbidities, and BMI ≥ 30 had a **73.8% probability** of transient decline (95% CI: 49.5–89.0%) versus 18.7% for those without risk factors (McDeed et al., 2024, *JNCI Cancer Spectrum*). The original finding by Ahles et al. (2003, *Psycho-Oncology*; n = 80) showed ε4 carriers scored significantly lower on visual memory and spatial ability (d ≈ 0.50). However, results are **inconsistent**: the CANTO-Cog study (n = 334; 19% APOE ε4+) found no significant overall association (2025, *Breast Cancer Research and Treatment*), suggesting the effect may be domain-specific or modified by treatment type.

**Baseline cognitive impairment** is a robust independent predictor. Dos Santos et al. (2021, *Cancers*) reported **OR = 2.58** (p = 0.022) for abnormal baseline MMSE predicting further cognitive impairment during chemotherapy in 310 elderly cancer patients (multivariate logistic regression). In the TLC LASSO model, baseline attention/processing/executive (APE) score was a selected predictor that improved model AUC from 0.71 to 0.75.

**Inflammatory markers** show the most promising quantitative effect sizes. Duivon et al. (2024, *Breast Cancer Research*; CANTO-Cog, n = 200) reported that **CRP > 3 mg/L predicted overall cognitive impairment at year 2 with OR = 2.84** (95% CI: 1.06–7.64, p = 0.04) and processing speed impairment with OR = 2.47 (95% CI: 1.05–5.87, p = 0.04). **Elevated IL-6 predicted episodic memory impairment with OR = 5.50** (95% CI: 1.43–36.60, p = 0.03)—the largest published effect size for any inflammatory marker, though with a wide confidence interval reflecting the small sample. Carroll et al. (2022, *Journal of Clinical Oncology*; TLC study) demonstrated that at CRP 3.0 mg/L, FACT-Cog scores were **9.5 points lower** than controls, increasing to **14.2 points lower** at CRP 10.0 mg/L, with effects independent of depression and anxiety. For **TNF-α**, soluble receptors show stronger associations than TNF-α itself: sTNFRI correlated **r = −0.65** (p = 0.001) with delayed match-to-sample performance (Williams et al., 2018, *Journal of Neuroimmunology*; n = 22), and sTNF-RII correlated **r = −0.55** (p = 0.02) with inferior frontal cortex metabolism (Ganz et al., 2013, *Brain, Behavior, and Immunity*; n = 93).

**Education and cognitive reserve** are strongly protective. Zhou et al. (2024, *Scientific Reports*) reported **OR = 0.103** for > 6 years education versus no education—a nearly 10-fold risk reduction. Amidi et al. (2015, *Psycho-Oncology*) found **OR = 0.87** (95% CI: 0.81–0.95, p < 0.01) per IQ point in 72 testicular cancer survivors. Ahles et al. (2010, *Journal of Clinical Oncology*) demonstrated a significant three-way interaction of cognitive reserve × chemotherapy × age (p < 0.003).

**Anxiety and depression** predict CICI through both direct and interaction effects. Zhou et al. (2024) reported **OR = 1.298 per unit increase** in anxiety score in their multivariate nomogram. Ramalho et al. (2017, *The Breast*; n = 418) found a significant interaction between baseline anxiety and chemotherapy on incident cognitive impairment (P for interaction = 0.028). Kim et al. (2019, *Journal of Korean Academy of Nursing*) found depression/anxiety explained 47.7% of CICI variance in structural equation modeling (β = 0.25, p = 0.002).

**Cumulative chemotherapy dose** shows a linear dose-response relationship. Collins et al. (2013, *Psycho-Oncology*; 60 patients + 60 controls) demonstrated that a linear model best fit the trajectory of cognitive decline across chemotherapy cycles for overall cognition, working memory, processing speed, and verbal memory—providing the most compelling clinical evidence for a causal dose-response relationship. Schagen et al. (2006) found dose-dependent impairment rates: high-dose 32%, CMF 20%, FEC 17%, no chemotherapy 6%.

| Risk factor | Best OR/HR/effect | Source | Analysis type |
|---|---|---|---|
| APOE ε4 carrier | Strongest LASSO predictor; d ≈ 0.50 | McDeed 2024; Ahles 2003 | Multivariate ML; between-group |
| Baseline MMSE abnormal | OR = 2.58 | Dos Santos 2021 | Multivariate logistic |
| CRP > 3 mg/L | OR = 2.84 (overall CI) | Duivon 2024 | Multivariate logistic |
| IL-6 elevated | OR = 5.50 (episodic memory) | Duivon 2024 | Multivariate logistic |
| sTNFRI | r = −0.65 | Williams 2018 | Adjusted regression |
| Education > 6 years | OR = 0.103 | Zhou 2024 | Multivariate logistic |
| Premorbid IQ per point | OR = 0.87 | Amidi 2015 | Logistic regression |
| Anxiety per unit | OR = 1.298 | Zhou 2024 | Multivariate logistic |
| CMF chemo vs. none | OR = 6.4 | Schagen 1999 | Cross-sectional comparison |
| Cumulative dose | Linear dose-response | Collins 2013 | Multilevel modeling |
| Subjective complaints | OR = 0.956 per PCI unit | Lange 2016 | Logistic regression |
| Age | Significant covariate (p < 0.05) | Multiple | No clean OR available |

---

## 3. Cohort demographics for realistic synthetic data generation

Constructing a realistic synthetic dataset for XGBoost training requires distributions anchored in published study characteristics. The typical CICI study enrolls **breast cancer patients with a mean age of 49–55 years** (SD ≈ 8–10). Janelsins et al. (2018, *Journal of Clinical Oncology*) reported mean age **53.4 years** for patients and 52.6 for controls (n = 580 + 363). Cheung et al. (2015, *Annals of Oncology*; Singapore) reported **50.5 ± 8.4 years** (n = 99). The TLC study enrolled older patients specifically, with ages 60–98 (Mandelblatt et al., 2018).

**APOE ε4 carrier prevalence** in breast cancer populations mirrors the general population at approximately **20–25%** in Caucasian cohorts. The UK Biobank breast cancer subsample (n = 2,876) showed **20% APOE ε4 carriers** (Lehrer & Rheinstein, 2021, *PMC*). The CANTO-Cog study reported **19%** ε4 carriers among 334 breast cancer patients (Duivon et al., 2024). General population ε4 allele frequency is approximately 13–15%.

**Baseline cognitive scores** in CICI studies typically use comprehensive neuropsychological batteries rather than screening tools. When screening tools are used: baseline MoCA averages approximately **25 ± 3** (Fernandes et al., 2017, reported median 24, IQR 21–26 in a Brazilian breast cancer cohort, n = 418), and baseline MMSE averages approximately **28 ± 2**. The MMSE has significant ceiling effects and low sensitivity (18%) for mild cognitive impairment versus MoCA (90% sensitivity).

**Education** varies substantially by geography. Western cohorts report approximately **14 ± 3 years** (Janelsins et al., 2017, 2018), while Asian cohorts report **10.8 ± 3.3 years** (Cheung et al., 2015, Singapore). Zhou et al. (2024) used categorical education (0, 1–6, > 6 years) in their Chinese nomogram cohort.

**Inflammatory marker distributions** at baseline in breast cancer patients show considerable variability across assay platforms.

| Marker | Baseline (pre-chemo) | Post-chemo change | Units | Key source |
|---|---|---|---|---|
| IL-6 | Median 1.5–2.5 | Significant increase (p < 0.0001) | pg/mL | Cheung 2015; Ganz 2013 |
| CRP | Median 3.0–3.5 | Variable; survivor-specific effects | mg/L | Carroll 2022; Duivon 2024 |
| TNF-α | Typically < 10 (high variability) | Transient increase | pg/mL | Ganz 2013; Williams 2018 |
| sTNF-RII | Elevated vs. controls | Declines over 12 months post-treatment | pg/mL | Ganz 2013 |

Important caveats for synthetic data: **14% of IL-6 values** and **37–83% of other cytokine values** (IL-1β, IL-2, IL-10, IFN-γ) fell below detection limits in Cheung et al. (2015). Distributions are typically right-skewed and require log-transformation. Ramalho et al. (2023, *MDPI*) reported TNF-α of 33.84 ± 36.18 pg/mL in a South African cohort, illustrating extreme assay-dependent variability.

**Cumulative cyclophosphamide doses** follow standard protocol ranges: AC regimen delivers **600 mg/m² × 4 cycles = 2,400 mg/m²** total; dose-dense AC uses the same cumulative dose on a 14-day cycle; classical oral CMF delivers approximately **8,400 mg/m²** total (100 mg/m²/day × 14 days × 6 cycles, ~75% oral bioavailability); FEC delivers **3,000–3,600 mg/m²** (500–600 mg/m² × 6 cycles IV).

**Cohort sizes** in the literature range from **n = 20–50** in pilot studies (Williams et al., 2018: n = 22), through **n = 50–200** in medium cohorts (Cheung et al., 2015: n = 99; Ganz et al., 2013: n = 93), to **n = 400–1,000** in large nationwide studies (Janelsins et al., 2017: n = 581 + 364; TLC: n = 344 + 347), and **n > 1,000** in very large cohorts (Boscher et al., 2020: n = 1,393). Meta-analyses have synthesized up to 52 studies (Whittaker et al., 2022).

---

## 4. The twelve most important papers for a CICI prediction model

The following papers represent the essential citation set for building a CICI risk prediction model, selected for methodological rigor, sample size, and relevance to predictive modeling.

**Foundational prospective studies.** Wefel et al. (2004, *Cancer*; n = 18) conducted the first prospective longitudinal CICI study, finding 33% pre-treatment impairment and 61% short-term post-chemotherapy decline—establishing the critical importance of baseline assessment. Ahles et al. (2002, *Journal of Clinical Oncology*; n = 128) demonstrated that chemotherapy-treated long-term survivors scored significantly lower than local-therapy controls (35% vs. 15% in impaired range). Janelsins et al. (2017, *Journal of Clinical Oncology*; n = 581 + 364) provided the largest prospective multicenter evidence of post-chemotherapy cognitive complaints. Mandelblatt et al. (2018, *Journal of Clinical Oncology*; TLC study; n = 344 + 347) showed chemotherapy effects were largely confined to APOE ε4 carriers in older patients (p = 0.03 for three-way interaction).

**Genetic moderator evidence.** Ahles et al. (2003, *Psycho-Oncology*; n = 80) was the seminal paper linking APOE ε4 to CICI vulnerability (d ≈ 0.50 for visual memory). The CANTO-Cog replication attempt (2025, *Breast Cancer Research and Treatment*; n = 334) found null overall results, highlighting the need for domain-specific and treatment-stratified analyses.

**Inflammatory biomarker studies.** Duivon et al. (2024, *Breast Cancer Research*; CANTO-Cog; n = 200) reported the strongest published ORs for inflammatory predictors: **CRP > 3 mg/L: OR = 2.84** for overall impairment; **IL-6: OR = 5.50** for episodic memory impairment. Carroll et al. (2022, *Journal of Clinical Oncology*; TLC) demonstrated dose-dependent CRP effects on self-reported cognition independent of mood. Ganz et al. (2013, *Brain, Behavior, and Immunity*; n = 93) linked sTNF-RII to frontal cortex metabolism and memory complaints.

**Predictive modeling papers.** McDeed et al. (2024, *JNCI Cancer Spectrum*; TLC) built the most rigorous CICI prediction model: LASSO using 94 predictors, achieving **AUC = 0.75** (transient decline) and **AUC = 0.79** (persistent decline). Key selected features: APOE ε4, HER2 status, obesity, cardiovascular comorbidities, baseline APE score. Zhou et al. (2024, *Scientific Reports*; n = 515) developed a nomogram with **AUC = 0.955** using education, anxiety, chemotherapy, benefit finding, and fasting glucose—though the unusually high AUC and cross-sectional design raise overfitting concerns.

**Meta-analyses and guidelines.** Whittaker et al. (2022, *Scientific Reports*; 52 studies) provided definitive prevalence estimates by assessment method. Wefel et al. (2011, *The Lancet Oncology*) established the ICCTF consensus criteria that standardize outcome measurement.

Collins et al. (2013, *Psycho-Oncology*; 60 + 60) provided the strongest evidence for a causal dose-response relationship between cumulative chemotherapy exposure and cognitive decline.

---

## 5. How to define CICI as a binary outcome variable

The **International Cognition and Cancer Task Force (ICCTF)** established consensus criteria in 2011 (Wefel et al., 2011, *The Lancet Oncology*): cognitive impairment is defined as **≥ 2 neuropsychological test scores falling ≤ 1.5 SD below the normative mean, OR ≥ 1 test score ≤ 2.0 SD below the mean**. This definition accounts for the multiple-comparison problem inherent in administering test batteries and has become the most widely cited standard. However, adoption remains incomplete—only **37.6%** of 93 post-2011 studies using neuropsychological tests incorporated the ICCTF definition, with adoption ranging from 0% (2011) to 56.2% (2022) (ASCO 2024 abstract).

The ICCTF recommended three **core neuropsychological tests**: the Hopkins Verbal Learning Test–Revised (HVLT-R) for learning and memory, the Trail Making Test Parts A and B (TMT) for attention, processing speed, and executive function, and the Controlled Oral Word Association Test (COWA) for verbal fluency. Supplementary tests commonly used include the Rey Auditory Verbal Learning Test (RAVLT), WAIS Digit Span and Digit Symbol-Coding, the Stroop Color-Word Test, and the Grooved Pegboard. The TLC study used six neuropsychological tests covering two composite domains: attention/processing/executive function (APE) and learning/memory (LM), z-scored against matched controls.

For **longitudinal binarization**, the Reliable Change Index (RCI) determines whether an individual's score change exceeds measurement error and practice effects: RCI = (X₂ − X₁ − practice effect) / SE_diff. A patient is classified as having "reliable cognitive decline" if RCI exceeds the critical threshold (typically |RCI| > 1.645 one-tailed). This approach is superior to cross-sectional cutoffs because it accounts for test-retest reliability and learning effects from repeated administration.

**Self-report binarization** uses the FACT-Cog Perceived Cognitive Impairments (PCI) subscale with a cutoff of **< 54** (18-item) or **< 60** (20-item), achieving **76% sensitivity and 82–84% specificity** against neuropsychological testing (Van Dyk et al., 2020, *JNCI Cancer Spectrum*). The full FACT-Cog has four subscales (PCI, Quality of Life Impact, Comments from Others, and Perceived Cognitive Abilities), with total scores from 0 to 148 (higher = better) and internal consistency α = 0.93–0.97 for PCI.

A critical consideration for model building: **subjective and objective measures often do not correlate**. Self-reported complaints are more strongly associated with psychological distress than with neuropsychological test performance. For an XGBoost model, the choice of outcome measure will fundamentally shape what the model predicts—objective impairment on testing versus patient-perceived dysfunction versus composite definitions.

---

## 6. No validated CICI risk calculator exists—but proof-of-concept models are emerging

An extensive search confirms that **no externally validated, clinical-grade CICI risk prediction tool** comparable to established instruments like the Framingham cardiovascular risk score or the CARG chemotherapy toxicity score exists in clinical practice. A 2025 comprehensive review (Duan et al., *Clinical and Experimental Medicine*) synthesizes the current landscape and confirms all published models remain in early development. This represents a significant clinical gap that a well-constructed XGBoost model could address.

The most methodologically rigorous model is the **TLC study LASSO** (McDeed et al., 2024, *JNCI Cancer Spectrum*), which evaluated 94 baseline predictors across clinical, genetic, cognitive, and inflammatory domains in older breast cancer survivors. It achieved **AUC = 0.75 for transient decline** and **AUC = 0.79 for persistent decline**, with APOE ε4, HER2 positivity, obesity, cardiovascular comorbidities, and baseline cognitive scores as selected features. This provides the most realistic performance benchmark for a CICI prediction model.

Other published models include Zhou et al. (2024, *Scientific Reports*; n = 515 breast cancer, **AUC = 0.955**) using education, anxiety, chemotherapy, and fasting glucose—though the cross-sectional design and unusually high AUC suggest overfitting. Liu et al. (2024, *Journal of Cancer Survivorship*; n = 741 breast cancer, **AUC = 0.882**) used education, exercise frequency, chemotherapy cycles, and fatigue, validated by 5-fold cross-validation. A colorectal cancer model (2021, *BioMed Research International*; n = 386) achieved **AUC = 0.799** with logistic regression using 14 predictors. All published models use **internal validation only**—none have been validated across independent populations or settings.

Related tools not specific to CICI include the **CARG score** (Cancer and Aging Research Group), which predicts grade 3–5 chemotherapy toxicity in older adults using 11 variables including cognition, and the **CRASH score**, which includes Mini-Mental Status examination. These demonstrate that cognition is already recognized as a component of chemotherapy risk assessment, supporting the clinical rationale for a dedicated CICI prediction tool.

---

## Bonus: brain imaging datasets for multimodal CICI research

**No publicly available neuroimaging dataset specifically designed for CICI** (combining chemotherapy patient brain scans with cognitive outcome labels) currently exists. This is a critical gap for developing imaging-based or multimodal prediction models. However, several adjacent datasets offer substantial value.

The **UK Biobank** represents the most promising resource for retrospective CICI neuroimaging research. With **> 40,000 brain scans** across six MRI modalities (T1, T2-FLAIR, susceptibility-weighted, diffusion, resting-state fMRI, task fMRI) linked to NHS health records including cancer diagnoses and treatment data, cognitive testing batteries, genetics (including APOE genotype), and blood biomarkers, researchers can potentially identify chemotherapy-exposed participants with both brain imaging and cognitive outcomes. Access requires an approved research application.

**ADNI** (Alzheimer's Disease Neuroimaging Initiative; adni.loni.usc.edu) provides a gold-standard framework for imaging-cognition research with **~2,000+ participants** across cognitively normal, MCI, and AD groups. It includes structural MRI, fMRI, DTI, PET (amyloid, tau, FDG), extensive neuropsychological testing (MMSE, ADAS-Cog, CDR), APOE genotype, CSF biomarkers, and blood markers. While not chemotherapy-specific, ADNI enables transfer learning for cognitive decline detection models and benchmarking of imaging biomarkers that may overlap with CICI (e.g., hippocampal volume changes, white matter integrity).

**OASIS-3** (Open Access Series of Imaging Studies; sites.wustl.edu/oasisbrains) offers **1,378 participants** with multimodal imaging (T1, T2, FLAIR, ASL, resting-state BOLD, DTI) plus **2,157+ PET scans**, CDR staging, MMSE, longitudinal cognitive tracking, and APOE status. Its open-access model and breadth of modalities make it particularly useful for developing imaging-based cognitive impairment classifiers.

The **BraTS dataset family** (Brain Tumor Segmentation Challenge; synapse.org/brats) provides expert-annotated brain tumor segmentation across **1,000+** cases with T1, T1-contrast, T2, and T2-FLAIR modalities. BraTS 2024 includes post-treatment glioma segmentation, meningioma (570 cases), and pediatric glioma (457 cases) sub-challenges. While BraTS contains no cognitive outcome labels, it offers brain segmentation transfer learning opportunities and understanding of treatment-related structural changes.

**TCIA** (The Cancer Imaging Archive; cancerimagingarchive.net) hosts several cancer neuroimaging collections including Brain-Tumor-Progression (20 GBM patients with longitudinal MRI post-chemoradiation), RIDER-NEURO-MRI (19 patients with repeat DCE-MRI and DTI), and brain metastasis datasets. None include cognitive outcomes, but they provide cancer-specific brain imaging for transfer learning. Chen et al. (2021, *Journal of Personalized Medicine*) achieved 80% accuracy classifying CICI from resting-state fMRI using SE-ResNet-50 and SE-DenseNet-121, but used a small private dataset (55 post-chemo survivors + 65 controls).

| Dataset | Chemo patients? | Cognitive labels? | Imaging modalities | Size | CICI relevance |
|---|---|---|---|---|---|
| UK Biobank | Via NHS linkage | ✅ | T1, T2, dMRI, fMRI, SWI | 40,000+ scans | Very high |
| ADNI | ❌ | ✅ Extensive | MRI, fMRI, DTI, PET | ~2,000+ | High |
| OASIS-3 | ❌ | ✅ CDR, MMSE | T1, T2, FLAIR, DTI, PET | 1,378 | High |
| BraTS 2024 | Post-treatment glioma | ❌ | T1, T1c, T2, FLAIR | 1,000+ | Moderate |
| TCIA | ✅ Brain tumors | ❌ | Various MRI | 20–500+ | Moderate |
| OpenNeuro | ❌ | Variable | Mostly fMRI | 600+ datasets | Low |

---

## Conclusion: a clear path from evidence to prediction

The CICI literature reveals a field with strong epidemiological evidence but a significant translational gap. Three findings particularly matter for XGBoost model development. First, **inflammatory markers (CRP and IL-6) offer the highest quantitative effect sizes** among modifiable, measurable biomarkers (ORs of 2.84 and 5.50 respectively from Duivon et al., 2024), making them high-priority features alongside APOE ε4 genotype and baseline cognitive reserve. Second, the **TLC LASSO model's AUC of 0.75–0.79** establishes a realistic performance ceiling for tabular prediction with current features—an XGBoost model with nonlinear interactions and gradient boosting could plausibly exceed this, particularly by capturing the documented interactions between age, cognitive reserve, and APOE status that linear models handle poorly. Third, the **ICCTF criteria (≥ 2 tests ≤ −1.5 SD or ≥ 1 test ≤ −2.0 SD)** provide a standardized binary outcome that, combined with the RCI method for longitudinal studies, enables consistent model training and evaluation across datasets.

The absence of an externally validated CICI risk calculator is not merely a gap—it is an opportunity. With synthetic data generation anchored in the cohort distributions documented here (mean age ~53, ~22% APOE ε4+, baseline MoCA ~25, IL-6 ~2 pg/mL, CRP ~3 mg/L), an XGBoost model incorporating genetic, inflammatory, cognitive, and psychological features could deliver clinically meaningful risk stratification for the estimated one-third of chemotherapy patients who will develop significant cognitive impairment.