"""
CICI Tabular Risk Model
========================
Literature-grounded XGBoost classifier for predicting chemotherapy-induced
cognitive impairment (CICI). All parameters sourced from published studies
via CICI Info/cici_config.py.

Pipeline:
  1. Generate synthetic cohort from published distributions
  2. Assign CICI labels probabilistically using published ORs
  3. Train XGBoost (primary) + Logistic Regression (baseline)
  4. Evaluate on held-out test set with bootstrap CIs
  5. Save poster-ready figures and metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "CICI Info"))
from cici_config import COHORT, BIOMARKERS, REGIMENS, REGIMEN_WEIGHTS, ODDS_RATIOS

OUT_DIR = PROJECT_ROOT / "outputs" / "cici_risk_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 1000


# ═══════════════════════════════════════════════════════════════════════
# 1. Synthetic cohort generation
# ═══════════════════════════════════════════════════════════════════════

def _sample_clipped_normal(rng, mean, sd, lo, hi, n):
    vals = rng.normal(mean, sd, n)
    return np.clip(vals, lo, hi)


def _sample_lognormal_clipped(rng, log_mean, log_sd, lo, hi, n):
    vals = rng.lognormal(log_mean, log_sd, n)
    return np.clip(vals, lo, hi)


def generate_cohort(n: int = None, seed: int = SEED) -> pd.DataFrame:
    n = n or COHORT["n_synthetic"]
    rng = np.random.default_rng(seed)

    regimen_names = list(REGIMEN_WEIGHTS.keys())
    regimen_probs = np.array([REGIMEN_WEIGHTS[r] for r in regimen_names])
    regimen_probs /= regimen_probs.sum()

    regimens = rng.choice(regimen_names, size=n, p=regimen_probs)

    a = COHORT["age"]
    age = _sample_clipped_normal(rng, a["mean"], a["sd"], a["min"], a["max"], n)

    e = COHORT["education_years"]
    education = _sample_clipped_normal(rng, e["mean"], e["sd"], e["min"], e["max"], n)

    apoe_e4 = rng.binomial(1, COHORT["apoe_e4_carrier_prob"]["value"], n)

    m = COHORT["baseline_moca"]
    moca = _sample_clipped_normal(rng, m["mean"], m["sd"], m["min"], m["max"], n)
    moca = np.round(moca).astype(int)

    # HADS-A anxiety: approx N(6, 3.5) clipped [0, 21] — typical cancer cohort
    anxiety = _sample_clipped_normal(rng, 6.0, 3.5, 0, 21, n)

    # Inflammatory biomarkers
    b = BIOMARKERS
    il6_bl = _sample_lognormal_clipped(
        rng, b["il6_baseline_pgml"]["log_mean"], b["il6_baseline_pgml"]["log_sd"],
        b["il6_baseline_pgml"]["min"], b["il6_baseline_pgml"]["max"], n)
    il6_pc = _sample_lognormal_clipped(
        rng, b["il6_postchemo_pgml"]["log_mean"], b["il6_postchemo_pgml"]["log_sd"],
        b["il6_postchemo_pgml"]["min"], b["il6_postchemo_pgml"]["max"], n)
    crp_bl = _sample_lognormal_clipped(
        rng, b["crp_baseline_mgl"]["log_mean"], b["crp_baseline_mgl"]["log_sd"],
        b["crp_baseline_mgl"]["min"], b["crp_baseline_mgl"]["max"], n)
    crp_pc = _sample_lognormal_clipped(
        rng, b["crp_postchemo_mgl"]["log_mean"], b["crp_postchemo_mgl"]["log_sd"],
        b["crp_postchemo_mgl"]["min"], b["crp_postchemo_mgl"]["max"], n)
    tnfa_bl = _sample_lognormal_clipped(
        rng, b["tnfa_baseline_pgml"]["log_mean"], b["tnfa_baseline_pgml"]["log_sd"],
        b["tnfa_baseline_pgml"]["min"], b["tnfa_baseline_pgml"]["max"], n)

    # Cumulative cyclophosphamide dose + n_cycles per regimen
    cyclo_dose = np.zeros(n, dtype=np.float32)
    n_cycles = np.zeros(n, dtype=np.int32)
    for i, reg in enumerate(regimens):
        info = REGIMENS[reg]
        n_cycles[i] = info["n_cycles"]
        dose_spec = info["cyclo_dose_mg_m2_total"]
        if dose_spec is not None:
            cyclo_dose[i] = float(np.clip(
                rng.normal(dose_spec["mean"], dose_spec["sd"]),
                dose_spec["min"], dose_spec["max"],
            ))

    # Controls get post-chemo biomarkers equal to baseline (no treatment)
    control_mask = np.isin(regimens, ["CONTROL_CANCER_NO_CHEMO", "CONTROL_HEALTHY"])
    il6_pc[control_mask] = il6_bl[control_mask]
    crp_pc[control_mask] = crp_bl[control_mask]

    df = pd.DataFrame({
        "subject_id": [f"SYN_{i:04d}" for i in range(n)],
        "regimen": regimens,
        "age": np.round(age, 1),
        "education_years": np.round(education, 1),
        "apoe_e4": apoe_e4,
        "baseline_moca": moca,
        "anxiety_score": np.round(anxiety, 1),
        "il6_baseline": np.round(il6_bl, 2),
        "il6_postchemo": np.round(il6_pc, 2),
        "crp_baseline": np.round(crp_bl, 2),
        "crp_postchemo": np.round(crp_pc, 2),
        "tnfa_baseline": np.round(tnfa_bl, 2),
        "n_cycles": n_cycles,
        "cumulative_cyclo_dose": np.round(cyclo_dose, 0),
    })
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Probabilistic CICI label assignment
# ═══════════════════════════════════════════════════════════════════════

def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def assign_cici_labels(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(df)

    base_incidences = df["regimen"].map(
        {k: v["cici_incidence"]["point"] for k, v in REGIMENS.items()}
    ).values
    log_odds = _logit(base_incidences)

    # Calibration offset: the additive OR contributions shift the marginal
    # rate above published levels. This intercept correction brings the
    # population-level rate back to ~35% (Whittaker 2022 meta-analysis:
    # 21-34% by neuropsych battery, 39.9% pooled by Gregoire 2025).
    log_odds -= 1.5

    # APOE e4: OR = 2.50
    log_odds += np.log(ODDS_RATIOS["apoe_e4_carrier"]["OR"]) * df["apoe_e4"].values

    # Baseline cognitive impairment (MoCA < 22): OR = 2.58
    cog_impaired = (df["baseline_moca"].values < 22).astype(float)
    log_odds += np.log(ODDS_RATIOS["baseline_cognitive_impairment"]["OR"]) * cog_impaired

    # CRP elevated (> 3 mg/L): OR = 2.84
    crp_high = (df["crp_postchemo"].values > 3.0).astype(float)
    log_odds += np.log(ODDS_RATIOS["crp_elevated"]["OR"]) * crp_high

    # IL-6 elevated (above cohort median): OR = 5.50 (Duivon 2024)
    # Wide CI (1.43-36.60) but largest published effect for any inflammatory marker
    il6_median = np.median(df["il6_postchemo"].values)
    il6_high = (df["il6_postchemo"].values > il6_median).astype(float)
    log_odds += np.log(ODDS_RATIOS["il6_elevated"]["OR"]) * il6_high

    # Education protective: OR = 0.87 per IQ-proxy point (Amidi 2015)
    # Operationalize as per year of education above population mean
    edu_centered = df["education_years"].values - 14.0
    log_odds += np.log(ODDS_RATIOS["education_per_iq_point"]["OR"]) * edu_centered

    # Anxiety: OR = 1.298 per unit on HADS-A scale (Zhou 2024)
    anxiety_centered = df["anxiety_score"].values - 6.0
    log_odds += np.log(ODDS_RATIOS["anxiety_per_unit"]["OR"]) * anxiety_centered

    # Age x APOE interaction: chemo effects confined to e4 carriers in older patients
    # (Mandelblatt 2018, p=0.03 for three-way interaction)
    age_centered = (df["age"].values - 53.4) / 9.0
    log_odds += 0.25 * age_centered * df["apoe_e4"].values

    # Cumulative dose: linear dose-response (Collins 2013)
    dose_normed = df["cumulative_cyclo_dose"].values / 8400.0
    log_odds += 0.5 * dose_normed

    prob = _sigmoid(log_odds)
    cici_label = rng.binomial(1, prob).astype(int)

    df = df.copy()
    df["cici_probability"] = np.round(prob, 4)
    df["cici_label"] = cici_label
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Feature matrix construction
# ═══════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "age", "education_years", "apoe_e4", "baseline_moca", "anxiety_score",
    "il6_baseline", "il6_postchemo", "crp_baseline", "crp_postchemo",
    "tnfa_baseline", "n_cycles", "cumulative_cyclo_dose",
]


def build_feature_matrix(df: pd.DataFrame):
    regimen_dummies = pd.get_dummies(df["regimen"], prefix="reg", dtype=float)
    X = pd.concat([df[FEATURE_COLS], regimen_dummies], axis=1)
    y = df["cici_label"].values
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, seed=SEED):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train, seed=SEED):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=seed, solver="lbfgs",
    )
    model.fit(X_scaled, y_train)
    return model, scaler


# ═══════════════════════════════════════════════════════════════════════
# 5. Evaluation with bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_metrics(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed + 99)
    results = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        results.append(compute_metrics(y_true[idx], y_prob[idx]))

    summary = {}
    for key in results[0]:
        vals = np.array([r[key] for r in results])
        summary[key] = {
            "mean": float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_roc_curves(y_test, xgb_prob, lr_prob, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    auc_xgb = roc_auc_score(y_test, xgb_prob)
    ax.plot(fpr_xgb, tpr_xgb, lw=2.2,
            label=f"XGBoost (AUC = {auc_xgb:.3f})", color="#2563eb")

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    auc_lr = roc_auc_score(y_test, lr_prob)
    ax.plot(fpr_lr, tpr_lr, lw=2.2, ls="--",
            label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="#dc2626")

    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("CICI Risk Prediction — ROC Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(xgb_model, X_test, save_path):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_test.columns.tolist()

    order = np.argsort(mean_abs_shap)
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = mean_abs_shap[order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(sorted_names) * 0.35)))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_vals)))
    ax.barh(range(len(sorted_vals)), sorted_vals, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Feature Importance — CICI XGBoost Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_test, xgb_prob, lr_prob, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    frac_xgb, mean_xgb = calibration_curve(y_test, xgb_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_xgb, frac_xgb, "o-", lw=2, label="XGBoost", color="#2563eb")

    frac_lr, mean_lr = calibration_curve(y_test, lr_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_lr, frac_lr, "s--", lw=2, label="Logistic Regression", color="#dc2626")

    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve — CICI Prediction", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_summary(df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    reg_counts = df["regimen"].value_counts()
    colors_reg = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
    ax.barh(reg_counts.index, reg_counts.values, color=colors_reg[:len(reg_counts)])
    ax.set_title("Regimen Distribution", fontweight="bold")
    ax.set_xlabel("Count")

    ax = axes[0, 1]
    for label, color in [(0, "#60a5fa"), (1, "#ef4444")]:
        subset = df[df["cici_label"] == label]["age"]
        ax.hist(subset, bins=25, alpha=0.6, color=color,
                label=f"CICI={label} (n={len(subset)})")
    ax.set_title("Age Distribution by CICI", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlabel("Age")

    ax = axes[0, 2]
    cici_by_reg = df.groupby("regimen")["cici_label"].mean().sort_values()
    ax.barh(cici_by_reg.index, cici_by_reg.values, color="#ef4444", alpha=0.7)
    ax.set_title("CICI Rate by Regimen", fontweight="bold")
    ax.set_xlabel("Proportion CICI+")
    ax.set_xlim(0, 1)

    ax = axes[1, 0]
    ax.hist(df["il6_postchemo"], bins=40, color="#8b5cf6", alpha=0.7)
    ax.set_title("Post-Chemo IL-6 (pg/mL)", fontweight="bold")
    ax.set_xlabel("IL-6")

    ax = axes[1, 1]
    ax.hist(df["crp_postchemo"], bins=40, color="#f59e0b", alpha=0.7)
    ax.set_title("Post-Chemo CRP (mg/L)", fontweight="bold")
    ax.set_xlabel("CRP")

    ax = axes[1, 2]
    apoe_cici = df.groupby("apoe_e4")["cici_label"].mean()
    labels = ["Non-carrier", "APOE \u03b54+"]
    vals = [apoe_cici.get(0, 0), apoe_cici.get(1, 0)]
    ax.bar(labels, vals, color=["#60a5fa", "#ef4444"], alpha=0.8, width=0.5)
    ax.set_title("CICI Rate by APOE \u03b54", fontweight="bold")
    ax.set_ylabel("Proportion CICI+")
    ax.set_ylim(0, 1)

    fig.suptitle("Synthetic CICI Cohort Summary (n={:,})".format(len(df)),
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("CICI Tabular Risk Model — Literature-Grounded Pipeline")
    print("=" * 60)

    # ── 1. Generate cohort ──────────────────────────────────────────
    print("\n[1/5] Generating synthetic cohort...")
    df = generate_cohort()
    df = assign_cici_labels(df)

    cici_rate = df["cici_label"].mean()
    print(f"  Subjects: {len(df)}")
    print(f"  CICI+ rate: {cici_rate:.1%} ({df['cici_label'].sum()}/{len(df)})")
    print(f"  Regimens: {df['regimen'].value_counts().to_dict()}")

    df.to_csv(OUT_DIR / "synthetic_cohort.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'synthetic_cohort.csv'}")

    # ── 2. Train/test split ─────────────────────────────────────────
    print("\n[2/5] Building features and splitting...")
    X, y = build_feature_matrix(df)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train CICI+ rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # ── 3. Cross-validation ─────────────────────────────────────────
    print("\n[3/5] 5-fold CV on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    xgb_tmp = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    xgb_cv = cross_val_score(xgb_tmp, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  XGBoost CV AUC: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")

    scaler_tmp = StandardScaler()
    X_train_scaled = scaler_tmp.fit_transform(X_train)
    lr_tmp = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr_cv = cross_val_score(lr_tmp, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  Logistic CV AUC: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}")

    # ── 4. Final training and test evaluation ───────────────────────
    print("\n[4/5] Training final models and evaluating on test set...")
    xgb_model = train_xgboost(X_train, y_train)
    lr_model, lr_scaler = train_logistic(X_train, y_train)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    lr_prob = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]

    xgb_metrics = compute_metrics(y_test, xgb_prob)
    lr_metrics = compute_metrics(y_test, lr_prob)
    print(f"  XGBoost  — AUC: {xgb_metrics['auc']:.4f} | "
          f"Sens: {xgb_metrics['sensitivity']:.3f} | "
          f"Spec: {xgb_metrics['specificity']:.3f} | "
          f"F1: {xgb_metrics['f1']:.3f}")
    print(f"  Logistic — AUC: {lr_metrics['auc']:.4f} | "
          f"Sens: {lr_metrics['sensitivity']:.3f} | "
          f"Spec: {lr_metrics['specificity']:.3f} | "
          f"F1: {lr_metrics['f1']:.3f}")

    print("\n  Bootstrap 95% CIs (XGBoost):")
    xgb_boot = bootstrap_metrics(y_test, xgb_prob)
    for metric, vals in xgb_boot.items():
        print(f"    {metric:>12s}: {vals['mean']:.4f} "
              f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]")

    print("\n  Bootstrap 95% CIs (Logistic Regression):")
    lr_boot = bootstrap_metrics(y_test, lr_prob)
    for metric, vals in lr_boot.items():
        print(f"    {metric:>12s}: {vals['mean']:.4f} "
              f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]")

    # ── 5. Save outputs ─────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    metrics_rows = []
    for name, point, boot in [
        ("XGBoost", xgb_metrics, xgb_boot),
        ("LogisticRegression", lr_metrics, lr_boot),
    ]:
        for metric_name in point:
            metrics_rows.append({
                "model": name,
                "metric": metric_name,
                "value": point[metric_name],
                "ci_lo": boot[metric_name]["ci_lo"],
                "ci_hi": boot[metric_name]["ci_hi"],
            })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    cv_df = pd.DataFrame({
        "model": ["XGBoost"] * 5 + ["LogisticRegression"] * 5,
        "fold": list(range(1, 6)) * 2,
        "cv_auc": list(xgb_cv) + list(lr_cv),
    })
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    xgb_model.save_model(str(OUT_DIR / "model_xgboost.json"))

    plot_roc_curves(y_test, xgb_prob, lr_prob, OUT_DIR / "roc_curves.png")
    print(f"  Saved: roc_curves.png")

    plot_feature_importance(xgb_model, X_test, OUT_DIR / "feature_importance.png")
    print(f"  Saved: feature_importance.png")

    plot_calibration(y_test, xgb_prob, lr_prob, OUT_DIR / "calibration_curve.png")
    print(f"  Saved: calibration_curve.png")

    plot_cohort_summary(df, OUT_DIR / "cohort_summary.png")
    print(f"  Saved: cohort_summary.png")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
