"""
CICI Survival — Synthetic Cohort Generator
==========================================
Generates a literature-anchored synthetic time-to-event cohort for the
CICI survival-analysis refactor.

Pipeline
--------
1. generate_covariates(n)      : sample patient baseline features
2. compute_linear_predictor(df): eta = sum of log(HR_k) * x_k (centered)
3. generate_event_times(df)    : inverse-CDF Weibull sampling under
                                 proportional hazards: T ~ Weibull(k, lambda*exp(-eta/k))
4. apply_censoring(df, sched)  : administrative (tau=24), uninformative dropout
                                 (10% per 6 mo), interval censoring (schedule A/B/C)
5. main()                      : orchestrator; returns train / test DataFrames
6. sanity_check_hr_recovery()  : fits Cox PH on simulated data; verifies that
                                 estimated HRs fall within pre-registered
                                 acceptable ranges. This is the bug-catch gate.

Design decisions are pre-registered in SURVIVAL_DESIGN.md. Do not change
HR point estimates or covariate codings without updating that document.

Recommended repo location: src/survival/survival_cohort.py
Dependencies: numpy, pandas, lifelines (>= 0.27 for sanity check)

Usage:
    python -m src.survival.survival_cohort          # generate + sanity check
    from src.survival.survival_cohort import main
    df_train, df_test = main()
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# Load the two config modules from "CICI Info/" (explicit paths so type
# checkers resolve imports without mutating PYTHONPATH).
# ──────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CICI_INFO = _REPO_ROOT / "CICI Info"
_CICI_CONFIG_PATH = _CICI_INFO / "cici_config.py"
_CICI_SURVIVAL_CONFIG_PATH = _CICI_INFO / "cici_survival_config.py"


def _load_cici_module(name: str, path: Path) -> types.ModuleType:
    if not path.is_file():
        raise ImportError(f"Missing config file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


try:
    cfg = _load_cici_module("cici_config", _CICI_CONFIG_PATH)
    scfg = _load_cici_module("cici_survival_config", _CICI_SURVIVAL_CONFIG_PATH)
except ImportError as exc:
    raise ImportError(
        f"Could not import cici_config or cici_survival_config. "
        f"Expected them under {_CICI_INFO}. Original error: {exc}"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────
# 1. GENERATE COVARIATES
# ──────────────────────────────────────────────────────────────────────────
def generate_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample n patients with all baseline covariates for the survival DGP.

    Reuses the marginal distributions from cici_config.py (demographics,
    biomarkers, regimens, doses) and adds the new survival-specific
    covariates from cici_survival_config.ADDITIONAL_COVARIATES
    (HADS-A, PHQ-9, FACIT-F, BMI, CVD, HER2).

    Returns
    -------
    pd.DataFrame with columns:
      [patient_id, age, education_years, apoe_e4, baseline_moca,
       il6_postchemo, crp_postchemo, hads_a, phq9, fatigue_severity,
       bmi, cardiovascular_comorbidity, her2_positive, regimen,
       cumulative_dose_mg_m2, n_cycles,
       il6_elevated, crp_elevated, moca_low, bmi_obese]
    """
    # ── Demographics (from cici_config.COHORT) ──
    age = rng.normal(cfg.COHORT["age"]["mean"],
                     cfg.COHORT["age"]["sd"], size=n)
    age = np.clip(age, cfg.COHORT["age"]["min"], cfg.COHORT["age"]["max"])

    education = rng.normal(cfg.COHORT["education_years"]["mean"],
                           cfg.COHORT["education_years"]["sd"], size=n)
    education = np.clip(education,
                        cfg.COHORT["education_years"]["min"],
                        cfg.COHORT["education_years"]["max"])

    apoe_e4 = rng.binomial(1, cfg.COHORT["apoe_e4_carrier_prob"]["value"],
                           size=n)

    baseline_moca = rng.normal(cfg.COHORT["baseline_moca"]["mean"],
                               cfg.COHORT["baseline_moca"]["sd"], size=n)
    baseline_moca = np.clip(baseline_moca,
                            cfg.COHORT["baseline_moca"]["min"],
                            cfg.COHORT["baseline_moca"]["max"])

    # ── Inflammatory biomarkers (lognormal, from cici_config.BIOMARKERS) ──
    il6_post = rng.lognormal(
        mean=cfg.BIOMARKERS["il6_postchemo_pgml"]["log_mean"],
        sigma=cfg.BIOMARKERS["il6_postchemo_pgml"]["log_sd"], size=n
    )
    il6_post = np.clip(il6_post,
                       cfg.BIOMARKERS["il6_postchemo_pgml"]["min"],
                       cfg.BIOMARKERS["il6_postchemo_pgml"]["max"])

    crp_post = rng.lognormal(
        mean=cfg.BIOMARKERS["crp_postchemo_mgl"]["log_mean"],
        sigma=cfg.BIOMARKERS["crp_postchemo_mgl"]["log_sd"], size=n
    )
    crp_post = np.clip(crp_post,
                       cfg.BIOMARKERS["crp_postchemo_mgl"]["min"],
                       cfg.BIOMARKERS["crp_postchemo_mgl"]["max"])

    # ── Psychological / fatigue (from cici_survival_config.ADDITIONAL_COVARIATES) ──
    hads = _truncated_normal(
        rng, scfg.ADDITIONAL_COVARIATES["hads_anxiety"], n
    )
    phq = _truncated_normal(
        rng, scfg.ADDITIONAL_COVARIATES["phq9_depression"], n
    )
    fatigue = _truncated_normal(
        rng, scfg.ADDITIONAL_COVARIATES["facit_fatigue_severity"], n
    )
    bmi = _truncated_normal(
        rng, scfg.ADDITIONAL_COVARIATES["bmi_kgm2"], n
    )
    cvd = rng.binomial(
        1,
        scfg.ADDITIONAL_COVARIATES["cardiovascular_prevalence"]["p"],
        size=n
    )
    her2 = rng.binomial(
        1,
        scfg.ADDITIONAL_COVARIATES["her2_positive_prevalence"]["p"],
        size=n
    )

    # ── Regimens (categorical, from cici_config.REGIMEN_WEIGHTS) ──
    # Extend with PLATINUM and TARGETED at small frequencies to exercise
    # those HRs in the sanity check.
    regimen_weights = {
        "AC_T": 0.42,
        "AC": 0.14,
        "FAC_FEC": 0.09,
        "CMF": 0.05,
        "PLATINUM": 0.03,
        "TARGETED": 0.05,
        "CONTROL_CANCER_NO_CHEMO": 0.22,
    }
    regimen_labels = list(regimen_weights.keys())
    regimen_probs = np.array(list(regimen_weights.values()))
    regimen_probs = regimen_probs / regimen_probs.sum()
    regimen = rng.choice(regimen_labels, size=n, p=regimen_probs)

    # ── Cumulative cyclophosphamide-equivalent dose by regimen ──
    cumulative_dose = np.zeros(n)
    n_cycles = np.zeros(n, dtype=int)
    for i, r in enumerate(regimen):
        if r in cfg.REGIMENS:
            dose_info = cfg.REGIMENS[r]["cyclo_dose_mg_m2_total"]
            if dose_info is not None:
                cumulative_dose[i] = np.clip(
                    rng.normal(dose_info["mean"], dose_info["sd"]),
                    dose_info["min"], dose_info["max"]
                )
            n_cycles[i] = cfg.REGIMENS[r]["n_cycles"]
        else:
            # PLATINUM, TARGETED — not cyclophosphamide-based, dose = 0
            cumulative_dose[i] = 0.0
            n_cycles[i] = 4

    df = pd.DataFrame({
        "patient_id":                  np.arange(n),
        "age":                         age,
        "education_years":             education,
        "apoe_e4":                     apoe_e4,
        "baseline_moca":               baseline_moca,
        "il6_postchemo":               il6_post,
        "crp_postchemo":               crp_post,
        "hads_a":                      hads,
        "phq9":                        phq,
        "fatigue_severity":            fatigue,
        "bmi":                         bmi,
        "cardiovascular_comorbidity":  cvd,
        "her2_positive":               her2,
        "regimen":                     regimen,
        "cumulative_dose_mg_m2":       cumulative_dose,
        "n_cycles":                    n_cycles,
    })

    # ── Derived binary indicators (thresholds from cici_survival_config) ──
    df["il6_elevated"] = (df["il6_postchemo"] > 5.0).astype(int)
    df["crp_elevated"] = (df["crp_postchemo"] > 3.0).astype(int)
    df["moca_low"]     = (df["baseline_moca"] < 22).astype(int)
    df["bmi_obese"]    = (df["bmi"] >= 30).astype(int)

    # Regimen dummies (CONTROL is reference; excluded)
    df["regimen_CMF"]        = (df["regimen"] == "CMF").astype(int)
    df["regimen_AC_or_ACT"]  = df["regimen"].isin(["AC", "AC_T"]).astype(int)
    df["regimen_FAC_FEC"]    = (df["regimen"] == "FAC_FEC").astype(int)
    df["regimen_PLATINUM"]   = (df["regimen"] == "PLATINUM").astype(int)
    df["regimen_TARGETED"]   = (df["regimen"] == "TARGETED").astype(int)

    return df


def _truncated_normal(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    """Sample from a truncated normal per a spec dict with keys mean/sd/min/max."""
    x = rng.normal(spec["mean"], spec["sd"], size=n)
    return np.clip(x, spec["min"], spec["max"])


# ──────────────────────────────────────────────────────────────────────────
# 2. LINEAR PREDICTOR
# ──────────────────────────────────────────────────────────────────────────
def compute_linear_predictor(df: pd.DataFrame, mean_center: bool = True) -> np.ndarray:
    """
    Compute eta_i = sum_k log(HR_k) * x_ik for each patient i.

    Continuous covariates whose literature HRs are defined "per unit from zero"
    (HADS-A, PHQ-9, fatigue, cumulative dose) are NOT centered inside this
    function — their raw values enter. Age and education are centered at
    reference values (50 yr, 14 yr) that are built into their HR specs.

    When mean_center=True (default), the final eta is shifted so that the
    population-average patient has eta = 0. This makes the Weibull scale
    lambda=22 from cici_survival_config apply to a *typical* cohort member
    rather than a hypothetical zero-covariate patient, and keeps simulated
    event times spread across the 0-24 month window rather than collapsing
    all events before the first assessment visit.

    Cox PH is invariant to additive shifts in the linear predictor (they are
    absorbed into the baseline hazard), so every hazard ratio in
    cici_survival_config.HAZARD_RATIOS is recoverable regardless of whether
    centering was applied — this is the reason the sanity check works.
    """
    hr = scfg.HAZARD_RATIOS
    eta = np.zeros(len(df))

    # Binary covariates
    eta += np.log(hr["apoe_e4"]["HR"])                    * df["apoe_e4"].values
    eta += np.log(hr["il6_elevated_postchemo"]["HR"])     * df["il6_elevated"].values
    eta += np.log(hr["crp_elevated"]["HR"])               * df["crp_elevated"].values
    eta += np.log(hr["baseline_moca_low"]["HR"])          * df["moca_low"].values
    eta += np.log(hr["cardiovascular_comorbidity"]["HR"]) * df["cardiovascular_comorbidity"].values
    eta += np.log(hr["bmi_obese"]["HR"])                  * df["bmi_obese"].values
    eta += np.log(hr["her2_positive"]["HR"])              * df["her2_positive"].values

    # Regimen dummies (CONTROL = reference)
    eta += np.log(hr["regimen_CMF"]["HR"])       * df["regimen_CMF"].values
    eta += np.log(hr["regimen_AC_or_ACT"]["HR"]) * df["regimen_AC_or_ACT"].values
    eta += np.log(hr["regimen_FAC_FEC"]["HR"])   * df["regimen_FAC_FEC"].values
    eta += np.log(hr["regimen_PLATINUM"]["HR"])  * df["regimen_PLATINUM"].values
    eta += np.log(hr["regimen_TARGETED"]["HR"])  * df["regimen_TARGETED"].values

    # Continuous covariates (centered at reference values)
    eta += np.log(hr["age_per_decade"]["HR"]) * (df["age"].values - 50) / 10.0
    eta += np.log(hr["education_per_year"]["HR"]) * (df["education_years"].values - 14.0)
    eta += np.log(hr["anxiety_per_unit"]["HR"]) * df["hads_a"].values
    eta += np.log(hr["depression_per_unit"]["HR"]) * df["phq9"].values
    eta += np.log(hr["fatigue_per_unit"]["HR"]) * df["fatigue_severity"].values
    eta += np.log(hr["cumulative_dose_per_1000"]["HR"]) * (df["cumulative_dose_mg_m2"].values / 1000.0)

    if mean_center:
        eta = eta - eta.mean()

    return eta


# ──────────────────────────────────────────────────────────────────────────
# 3. EVENT-TIME GENERATION (Weibull with proportional hazards)
# ──────────────────────────────────────────────────────────────────────────
def generate_event_times(
    eta: np.ndarray,
    rng: np.random.Generator,
    hazard: str = "weibull_primary",
) -> np.ndarray:
    """
    Sample exact continuous event times under proportional hazards.

    For Weibull baseline h0(t) = (k/lambda) * (t/lambda)^(k-1), the
    hazard h(t|x) = h0(t) * exp(eta) is equivalent to a Weibull with
    scale lambda * exp(-eta/k) and the same shape k. Inverse CDF sampling:

        U ~ Uniform(0, 1)
        T = lambda * exp(-eta/k) * (-ln U)^(1/k)
    """
    if hazard == "weibull_primary":
        spec = scfg.BASELINE_HAZARD["weibull_primary"]
        k = spec["shape_k"]
        lam = spec["scale_lambda_months"]
    elif hazard == "log_logistic_alternative":
        spec = scfg.BASELINE_HAZARD["log_logistic_alternative"]
        return _sample_log_logistic(eta, rng, spec["scale_alpha_months"],
                                    spec["shape_beta"])
    else:
        raise ValueError(f"Unknown hazard distribution: {hazard}")

    U = rng.uniform(size=len(eta))
    T = lam * np.exp(-eta / k) * (-np.log(U)) ** (1.0 / k)
    return T


def _sample_log_logistic(eta, rng, alpha, beta):
    """Log-logistic AFT parameterization under proportional odds."""
    U = rng.uniform(size=len(eta))
    # Acceleration factor via AFT-like time scaling
    scale = alpha * np.exp(-eta / beta)
    return scale * ((1 - U) / U) ** (1.0 / beta)


# ──────────────────────────────────────────────────────────────────────────
# 4. CENSORING (administrative + uninformative dropout + interval)
# ──────────────────────────────────────────────────────────────────────────
def apply_censoring(
    exact_event_times: np.ndarray,
    eta: np.ndarray,
    rng: np.random.Generator,
    schedule: str = "B_cici_typical",
    informative: bool = False,
) -> pd.DataFrame:
    """
    Apply administrative, dropout, and interval censoring.

    Returns a DataFrame with columns:
      [T_exact, T_observed, event, interval_L, interval_R, dropout]

    T_observed is the time recorded for analysis (midpoint-imputed for
    interval censoring). interval_L / interval_R bracket the true event
    time for interval-censored likelihoods.
    """
    n = len(exact_event_times)
    tau = scfg.CENSORING["administrative_horizon_months"]

    # Dropout times (uninformative by default)
    base_rate = scfg.CENSORING["uninformative_dropout_hazard_per_month"]
    if informative:
        rate = base_rate * np.exp(
            scfg.CENSORING["informative_dropout_multiplier"] * eta
        )
    else:
        rate = np.full(n, base_rate)
    dropout_times = rng.exponential(scale=1.0 / rate, size=n)

    # Administrative censoring at tau
    censor_times = np.minimum(dropout_times, tau)

    # Visit schedule
    visits = np.array(scfg.ASSESSMENT_SCHEDULES[schedule]["visit_months"])

    T_observed = np.zeros(n)
    event = np.zeros(n, dtype=int)
    interval_L = np.zeros(n)
    interval_R = np.zeros(n)

    for i in range(n):
        T_true = exact_event_times[i]
        C = censor_times[i]

        if T_true > C:
            # Censored before event detected
            event[i] = 0
            T_observed[i] = C
            interval_L[i] = C
            interval_R[i] = np.inf
        else:
            # Event occurs; detect at first visit >= T_true that also <= C
            visits_after = visits[visits >= T_true]
            visits_after = visits_after[visits_after <= C]
            if len(visits_after) == 0:
                # Censored: would have been detected after dropout / horizon
                event[i] = 0
                T_observed[i] = C
                interval_L[i] = C
                interval_R[i] = np.inf
            else:
                detection = visits_after[0]
                # Interval = [previous visit, detection]
                visits_before = visits[visits < T_true]
                prev = visits_before[-1] if len(visits_before) > 0 else 0.0
                event[i] = 1
                # Midpoint imputation for standard Cox
                T_observed[i] = (prev + detection) / 2.0
                interval_L[i] = prev
                interval_R[i] = detection

    return pd.DataFrame({
        "T_exact":     exact_event_times,
        "T_observed":  T_observed,
        "event":       event,
        "interval_L":  interval_L,
        "interval_R":  interval_R,
        "dropout_time": dropout_times,
    })


# ──────────────────────────────────────────────────────────────────────────
# 5. ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────
def main(
    n: int | None = None,
    seed: int | None = None,
    schedule: str = "B_cici_typical",
    hazard: str = "weibull_primary",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full synthetic CICI survival cohort and return train / test
    splits ready for lifelines / sksurv.

    Returns
    -------
    df_train, df_test : pd.DataFrame
        Stratified 80/20 split on event status.
    """
    n = n or scfg.SURVIVAL_COHORT["n"]
    seed = seed or scfg.SURVIVAL_COHORT["seed"]
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[survival_cohort] n={n}  seed={seed}  schedule={schedule}  hazard={hazard}")

    # Step 1 — covariates
    df = generate_covariates(n, rng)
    if verbose:
        print(f"[survival_cohort] covariates generated: {df.shape}")

    # Step 2 — linear predictor
    eta = compute_linear_predictor(df)
    df["eta"] = eta
    if verbose:
        print(f"[survival_cohort] eta: mean={eta.mean():.3f}  "
              f"sd={eta.std():.3f}  range=[{eta.min():.2f}, {eta.max():.2f}]")

    # Step 3 — exact event times
    T_exact = generate_event_times(eta, rng, hazard=hazard)
    if verbose:
        print(f"[survival_cohort] exact event times: "
              f"median={np.median(T_exact):.1f} mo  "
              f"frac<12mo={np.mean(T_exact < 12):.3f}  "
              f"frac<24mo={np.mean(T_exact < 24):.3f}")

    # Step 4 — censoring
    cens = apply_censoring(T_exact, eta, rng, schedule=schedule)
    df = pd.concat([df, cens], axis=1)
    if verbose:
        print(f"[survival_cohort] after censoring: "
              f"event rate = {df['event'].mean():.3f}  "
              f"median T_obs = {df['T_observed'].median():.1f} mo")

    # Check: 12-month event rate for reference patients (no chemo, APOE-)
    ref_mask = (df["regimen"] == "CONTROL_CANCER_NO_CHEMO") & (df["apoe_e4"] == 0)
    ref_12mo_rate = df.loc[ref_mask, "event"].mean() if ref_mask.sum() > 20 else np.nan
    if verbose:
        print(f"[survival_cohort] reference (control, APOE-) 12mo-ish event rate: {ref_12mo_rate:.3f}")
        target = scfg.REFERENCE_PATIENT["target_12mo_event_rate"]
        tol = scfg.REFERENCE_PATIENT["target_tolerance"]
        in_range = abs(ref_12mo_rate - target) <= tol if not np.isnan(ref_12mo_rate) else False
        print(f"[survival_cohort] target={target} +/- {tol}  "
              f"IN RANGE: {in_range}")

    # Step 5 — stratified 80/20 split
    rng_split = np.random.default_rng(seed + 1)
    idx_pos = df.index[df["event"] == 1].to_numpy()
    idx_neg = df.index[df["event"] == 0].to_numpy()
    idx_pos = idx_pos.copy()
    idx_pos = rng_split.permutation(idx_pos)
    idx_neg = idx_neg.copy()
    idx_neg = rng_split.permutation(idx_neg)
    split_pos = int(0.8 * len(idx_pos))
    split_neg = int(0.8 * len(idx_neg))
    train_idx = np.concatenate([idx_pos[:split_pos], idx_neg[:split_neg]])
    test_idx  = np.concatenate([idx_pos[split_pos:], idx_neg[split_neg:]])
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    if verbose:
        print(f"[survival_cohort] train: n={len(df_train)}  "
              f"event rate={df_train['event'].mean():.3f}")
        print(f"[survival_cohort] test:  n={len(df_test)}  "
              f"event rate={df_test['event'].mean():.3f}")

    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────────
# 6. HR RECOVERY SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────
def sanity_check_hr_recovery(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame | None:
    """
    Fit a Cox PH model on the simulated cohort and compare estimated HRs to
    the pre-registered acceptable ranges in cici_survival_config.RECOVERY_TOLERANCES.

    Failure = the simulation is buggy. Passing this check is the gate for
    moving to Sprint 2.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        warnings.warn(
            "lifelines not installed. Skipping HR recovery check. "
            "Install with: pip install lifelines"
        )
        return None

    # Build design matrix (all covariates in RECOVERY_TOLERANCES)
    design = pd.DataFrame({
        "apoe_e4":                    df["apoe_e4"].astype(float),
        "age_per_decade":              (df["age"] - 50) / 10.0,
        "education_per_year":          df["education_years"] - 14.0,
        "baseline_moca_low":           df["moca_low"].astype(float),
        "il6_elevated_postchemo":      df["il6_elevated"].astype(float),
        "crp_elevated":                df["crp_elevated"].astype(float),
        "anxiety_per_unit":            df["hads_a"].astype(float),
        "depression_per_unit":         df["phq9"].astype(float),
        "fatigue_per_unit":            df["fatigue_severity"].astype(float),
        "regimen_CMF":                 df["regimen_CMF"].astype(float),
        "regimen_AC_or_ACT":            df["regimen_AC_or_ACT"].astype(float),
        "regimen_FAC_FEC":              df["regimen_FAC_FEC"].astype(float),
        "regimen_PLATINUM":             df["regimen_PLATINUM"].astype(float),
        "regimen_TARGETED":             df["regimen_TARGETED"].astype(float),
        "cumulative_dose_per_1000":    df["cumulative_dose_mg_m2"] / 1000.0,
        "her2_positive":               df["her2_positive"].astype(float),
        "cardiovascular_comorbidity":  df["cardiovascular_comorbidity"].astype(float),
        "bmi_obese":                   df["bmi_obese"].astype(float),
    })
    design["T"] = df["T_observed"].values
    design["E"] = df["event"].values

    cph = CoxPHFitter(penalizer=0.001)  # tiny ridge for numerical stability
    cph.fit(design, duration_col="T", event_col="E", show_progress=False)

    results = []
    for covar, tol in scfg.RECOVERY_TOLERANCES.items():
        if covar not in cph.hazard_ratios_.index:
            continue
        est = float(cph.hazard_ratios_[covar])
        lo, hi = tol["acceptable"]
        passed = lo <= est <= hi
        results.append({
            "covariate":    covar,
            "true_HR":      tol["true_HR"],
            "estimated_HR": round(est, 3),
            "acceptable":   f"[{lo:.2f}, {hi:.2f}]",
            "passed":       passed,
        })
    result_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 72)
        print("HR RECOVERY SANITY CHECK")
        print("=" * 72)
        pass_rate = result_df["passed"].mean()
        print(result_df.to_string(index=False))
        print(f"\nOverall pass rate: {result_df['passed'].sum()}/{len(result_df)} "
              f"({pass_rate:.1%})")
        if pass_rate < 0.85:
            print("\n[WARNING] Pass rate below 85%. Likely simulation bug. "
                  "Do NOT proceed to Sprint 2 until resolved.")
        else:
            print("\n[OK] Sanity check passed. Safe to proceed to Sprint 2.")

    return result_df


# ──────────────────────────────────────────────────────────────────────────
# 7. CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_train, df_test = main()

    # Save cohort to disk for downstream models
    output_dir = _REPO_ROOT / "outputs" / "cici_survival"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_dir / "cohort_train.csv", index=False)
    df_test.to_csv(output_dir / "cohort_test.csv", index=False)
    print(f"\n[survival_cohort] saved cohorts to {output_dir}")

    # Run HR recovery sanity check on full cohort (train + test combined)
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    recovery = sanity_check_hr_recovery(df_full)
    if recovery is not None:
        recovery.to_csv(output_dir / "hr_recovery_check.csv", index=False)
        print(f"[survival_cohort] saved HR recovery report to "
              f"{output_dir / 'hr_recovery_check.csv'}")