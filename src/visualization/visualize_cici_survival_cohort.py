"""
CICI synthetic survival cohort — QA and exploratory figures
==========================================================
Reads `outputs/cici_survival/cohort_train.csv` and `cohort_test.csv` (or paths
you pass) and writes PNGs under `outputs/cici_survival/figures/`.

Designed for the literature-anchored DGP in `src/survival/survival_cohort.py`.
Optional: `hr_recovery_check.csv` for a Cox HR recovery forest plot.

Usage
-----
    python -m src.visualization.visualize_cici_survival_cohort
    python -m src.visualization.visualize_cici_survival_cohort --out-dir outputs/cici_survival/figures
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CICI_INFO = _REPO_ROOT / "CICI Info"
_SURVIVAL_CFG = _CICI_INFO / "cici_survival_config.py"


def _load_cici_survival_config() -> types.ModuleType | None:
    """Load config from repo path so static analyzers do not need CICI Info on PYTHONPATH."""
    if not _SURVIVAL_CFG.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "cici_survival_config", _SURVIVAL_CFG
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("cici_survival_config", mod)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


scfg = _load_cici_survival_config()

from lifelines import KaplanMeierFitter, NelsonAalenFitter


DEFAULT_TRAIN = _REPO_ROOT / "outputs" / "cici_survival" / "cohort_train.csv"
DEFAULT_TEST = _REPO_ROOT / "outputs" / "cici_survival" / "cohort_test.csv"
DEFAULT_HR_CSV = _REPO_ROOT / "outputs" / "cici_survival" / "hr_recovery_check.csv"


def _load_cohort(train_path: Path, test_path: Path) -> pd.DataFrame:
    parts = []
    for p, label in ((train_path, "train"), (test_path, "test")):
        if not p.exists():
            raise FileNotFoundError(f"Missing cohort CSV: {p}")
        d = pd.read_csv(p)
        d["_split"] = label
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _km_cumulative_incidence(kmf: KaplanMeierFitter, times: np.ndarray) -> dict[float, float]:
    """1 - S(t): crude cumulative incidence of the modeled event by t."""
    out = {}
    for t in times:
        try:
            s = float(kmf.predict(t))
        except Exception:
            s = float(kmf.survival_function_at_times(t).iloc[0])
        out[float(t)] = 1.0 - s
    return out


def plot_km_overall(df: Path | pd.DataFrame, out_path: Path) -> None:
    if isinstance(df, Path):
        d = pd.read_csv(df)
    else:
        d = df
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter(label="Overall (observed times)")
    kmf.fit(d["T_observed"], d["event"])
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Months on study")
    ax.set_ylabel("Survival (no CICI event)")
    ax.set_title("Kaplan–Meier — full synthetic cohort (train + test)")
    ax.set_xlim(0, 26)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_km_by_column(
    d: pd.DataFrame,
    column: str,
    out_path: Path,
    title: str,
    max_categories: int = 12,
) -> None:
    cats = d[column].astype(str).unique()
    if len(cats) > max_categories:
        vc = d[column].value_counts()
        cats = vc.head(max_categories).index.astype(str).tolist()
        d = d[d[column].astype(str).isin(cats)]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(cats), 2)))

    for i, cat in enumerate(sorted(cats, key=str)):
        sub = d[d[column].astype(str) == cat]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter(label=f"{cat} (n={len(sub)})")
        kmf.fit(sub["T_observed"], sub["event"])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=cmap[i % 10])

    ax.set_xlabel("Months on study")
    ax.set_ylabel("Survival (no CICI event)")
    ax.set_title(title)
    ax.set_xlim(0, 26)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nelson_aalen(d: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    naf = NelsonAalenFitter()
    naf.fit(d["T_observed"], d["event"])
    naf.plot_cumulative_hazard(ax=ax, label="Cumulative hazard Λ(t)")
    ax.set_xlabel("Months on study")
    ax.set_ylabel("Cumulative hazard")
    ax.set_title("Nelson–Aalen cumulative hazard — overall cohort")
    ax.set_xlim(0, 26)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_distributions(d: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.hist(d.loc[d["event"] == 1, "T_exact"], bins=40, alpha=0.7, label="Event (latent T_exact)", color="#c62828")
    ax.hist(d.loc[d["event"] == 0, "T_exact"], bins=40, alpha=0.5, label="Censored (T_exact)", color="#1565c0")
    ax.set_xlabel("T_exact (months)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Latent event times vs censoring status")

    ax = axes[0, 1]
    ax.hist(d.loc[d["event"] == 1, "T_observed"], bins=40, alpha=0.7, label="Event", color="#c62828")
    ax.hist(d.loc[d["event"] == 0, "T_observed"], bins=40, alpha=0.5, label="Censored", color="#1565c0")
    ax.set_xlabel("T_observed (months, midpoint if interval-detected)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Observed analysis times")

    ax = axes[1, 0]
    if (d["event"] == 1).any():
        delay = d.loc[d["event"] == 1, "T_observed"] - d.loc[d["event"] == 1, "T_exact"]
        ax.hist(delay.clip(-0.5, 12), bins=36, color="#6a1b9a", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("T_observed − T_exact (months), events only")
        ax.set_ylabel("Count")
        ax.set_title("Discretization / interval midpoint offset (events)")

    ax = axes[1, 1]
    width = (d["event"] == 1) & (d["interval_R"] < 1e9)
    if width.any():
        iw = d.loc[width, "interval_R"] - d.loc[width, "interval_L"]
        ax.hist(iw, bins=np.arange(0, 13) + 0.5, color="#2e7d32", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Interval width (months)")
        ax.set_ylabel("Count")
        ax.set_title("Detection interval width (events with finite R)")
    else:
        ax.text(0.5, 0.5, "No finite interval_R", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Time axis diagnostics (synthetic DGP)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regimen_counts(d: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vc = d["regimen"].value_counts().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vc)))
    vc.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Patients")
    ax.set_title("Regimen mix (full cohort)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eta_vs_texact(d: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ev = d["event"] == 1
    ax.scatter(
        d.loc[~ev, "eta"],
        d.loc[~ev, "T_exact"],
        s=8,
        alpha=0.35,
        c="#1565c0",
        label="Censored (latent T)",
    )
    ax.scatter(
        d.loc[ev, "eta"],
        d.loc[ev, "T_exact"],
        s=10,
        alpha=0.5,
        c="#c62828",
        label="Event by follow-up",
    )
    ax.set_xlabel("η (centered linear predictor)")
    ax.set_ylabel("T_exact (months)")
    ax.set_title("PH structure check: higher η → earlier events (stochastic)")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_covariate_marginals(d: pd.DataFrame, out_path: Path) -> None:
    cols = [
        ("age", "Age (y)"),
        ("education_years", "Education (y)"),
        ("baseline_moca", "MoCA"),
        ("hads_a", "HADS-A"),
        ("phq9", "PHQ-9"),
        ("fatigue_severity", "FACIT-F severity"),
        ("bmi", "BMI"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for i, (c, lab) in enumerate(cols):
        ax = axes[i]
        ax.hist(d[c].dropna(), bins=35, color="#37474f", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(lab)
        ax.set_ylabel("Count")
    axes[-1].axis("off")
    fig.suptitle("Baseline continuous covariates (simulated marginals)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binary_prevalence(d: pd.DataFrame, out_path: Path) -> None:
    keys = [
        ("apoe_e4", "APOE ε4"),
        ("il6_elevated", "IL-6 elevated"),
        ("crp_elevated", "CRP elevated"),
        ("moca_low", "MoCA < 22"),
        ("bmi_obese", "BMI ≥ 30"),
        ("cardiovascular_comorbidity", "CVD"),
        ("her2_positive", "HER2+"),
    ]
    labels, vals = [], []
    for col, name in keys:
        if col in d.columns:
            labels.append(name)
            vals.append(float(d[col].mean()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#455a64")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prevalence")
    ax.set_title("Binary covariate prevalence")
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 0.95), yi, f"{v:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_comparison(d: pd.DataFrame, out_path: Path) -> None:
    """Compare KM-based cumulative incidence at 12 / 24 mo to pre-registered targets."""
    horizons = np.array([12.0, 24.0])
    groups = [
        ("Full cohort", d),
        (
            "Reference: control, APOE−",
            d[(d["regimen"] == "CONTROL_CANCER_NO_CHEMO") & (d["apoe_e4"] == 0)],
        ),
        ("Chemo any (non-control)", d[d["regimen"] != "CONTROL_CANCER_NO_CHEMO"]),
    ]

    rows = []
    for name, sub in groups:
        if len(sub) < 30:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["T_observed"], sub["event"])
        ci = _km_cumulative_incidence(kmf, horizons)
        rows.append({"group": name, "n": len(sub), "ci_12": ci[12.0], "ci_24": ci[24.0]})

    summary = pd.DataFrame(rows)
    if summary.empty:
        return

    primary_12 = primary_24 = None
    if scfg is not None and hasattr(scfg, "OUTCOME_SURVIVAL"):
        primary_12 = scfg.OUTCOME_SURVIVAL["primary"]["expected_event_rate_12mo"]
        primary_24 = scfg.OUTCOME_SURVIVAL["primary"]["expected_event_rate_24mo"]
        tol = scfg.OUTCOME_SURVIVAL["primary"]["tolerance"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary["ci_12"], width=w, label="Observed ~CI by 12 mo (1−S(12))")
    ax.bar(x + w / 2, summary["ci_24"], width=w, label="Observed ~CI by 24 mo (1−S(24))")
    if primary_12 is not None:
        ax.axhline(primary_12, color="#e65100", linestyle="--", linewidth=1.2, label=f"Target primary 12 mo ({primary_12:.0%})")
        ax.axhline(primary_24, color="#bf360c", linestyle=":", linewidth=1.2, label=f"Target primary 24 mo ({primary_24:.0%})")
        if tol is not None:
            ax.fill_between([-0.5, len(summary) - 0.5], primary_12 - tol, primary_12 + tol, color="#ffe0b2", alpha=0.5, linewidth=0)
    ax.set_xticks(x, summary["group"], rotation=15, ha="right")
    ax.set_ylabel("Cumulative incidence (KM, midpoint times)")
    ax.set_ylim(0, min(1.05, max(summary["ci_24"].max(), 0.55) * 1.15))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Event burden vs design targets (RCI-primary rate bands from config)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hr_recovery_forest(hr_csv: Path, out_path: Path) -> bool:
    if not hr_csv.exists():
        return False
    hr = pd.read_csv(hr_csv).reset_index(drop=True)
    if hr.empty or "estimated_HR" not in hr.columns:
        return False

    def parse_acceptable(s) -> tuple[float, float] | None:
        if isinstance(s, str):
            m = re.findall(r"[\d.]+", s)
            if len(m) >= 2:
                return float(m[0]), float(m[1])
        return None

    n = len(hr)
    y_idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, max(4.5, n * 0.32)))

    est = hr["estimated_HR"].astype(float).to_numpy()
    true_hr = hr["true_HR"].astype(float).to_numpy() if "true_HR" in hr.columns else np.full(n, np.nan)

    for i in range(n):
        row = hr.iloc[i]
        lo_hi = parse_acceptable(row.get("acceptable"))
        passed = bool(row["passed"]) if "passed" in hr.columns and pd.notna(row["passed"]) else True
        color = "#2e7d32" if passed else "#c62828"
        if lo_hi:
            ax.plot([lo_hi[0], lo_hi[1]], [i, i], color="#bdbdbd", linewidth=5, solid_capstyle="butt", zorder=1)
        ax.scatter(est[i], i, color=color, s=42, zorder=3, edgecolors="white", linewidths=0.6, label=None)
        if not np.isnan(true_hr[i]):
            ax.scatter(true_hr[i], i, marker="D", s=22, color="k", zorder=4, edgecolors="white", linewidths=0.3)

    ax.set_yticks(y_idx, hr["covariate"].astype(str))
    ax.set_xlabel("Hazard ratio")
    ax.set_title("Cox PH recovery on synthetic cohort (◇ DGP HR, ● fitted HR)")
    ax.axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="k", markersize=7, label="DGP true HR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32", markersize=8, label="Fitted (pass)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c62828", markersize=8, label="Fitted (fail)"),
        Line2D([0], [0], color="#bdbdbd", linewidth=5, label="Acceptable range"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_km_train_test(d: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"train": "#1565c0", "test": "#c62828"}
    for split in ("train", "test"):
        sub = d[d["_split"] == split]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter(label=f"{split} (n={len(sub)})")
        kmf.fit(sub["T_observed"], sub["event"])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(split, None))
    ax.set_xlabel("Months on study")
    ax.set_ylabel("Survival (no CICI event)")
    ax.set_title("Train vs test KM (stratified split — curves should overlap)")
    ax.set_xlim(0, 26)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_metrics(d: pd.DataFrame, out_txt: Path) -> None:
    lines = []
    lines.append("CICI synthetic survival cohort — summary metrics")
    lines.append(f"n_total={len(d)}  train={(d['_split']=='train').sum()}  test={(d['_split']=='test').sum()}")
    lines.append(f"event_rate={d['event'].mean():.4f}")
    lines.append(f"T_observed: median={d['T_observed'].median():.3f}  mean={d['T_observed'].mean():.3f}")
    lines.append(f"T_exact (events): median={d.loc[d['event']==1,'T_exact'].median():.3f}")

    kmf = KaplanMeierFitter()
    kmf.fit(d["T_observed"], d["event"])
    ci = _km_cumulative_incidence(kmf, np.array([6.0, 12.0, 18.0, 24.0]))
    lines.append("KM cumulative incidence (1−S(t)), overall:")
    for t, v in sorted(ci.items()):
        lines.append(f"  t={t:.0f} mo: {v:.4f}")

    ref = d[(d["regimen"] == "CONTROL_CANCER_NO_CHEMO") & (d["apoe_e4"] == 0)]
    if len(ref) >= 20:
        lines.append(
            f"Reference subgroup n={len(ref)} (control, APOE−): "
            f"mean(event)={ref['event'].mean():.4f} "
            "(matches survival_cohort calibration printout; not the same as KM CI at 12 mo)"
        )
        kmr = KaplanMeierFitter()
        kmr.fit(ref["T_observed"], ref["event"])
        cir = _km_cumulative_incidence(kmr, np.array([12.0, 24.0]))
        lines.append("  KM cumulative incidence (1−S(t)) for that subgroup:")
        for t, v in sorted(cir.items()):
            lines.append(f"    t={t:.0f} mo: {v:.4f}")
        if scfg is not None:
            tgt = scfg.REFERENCE_PATIENT["target_12mo_event_rate"]
            tol = scfg.REFERENCE_PATIENT["target_tolerance"]
            lines.append(f"  REFERENCE_PATIENT target (vs mean event): {tgt:.2f} ± {tol:.2f}")

    lines.append("")
    lines.append("Regimen counts:")
    for k, v in d["regimen"].value_counts().items():
        lines.append(f"  {k}: {v}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize CICI synthetic survival cohort CSVs.")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "cici_survival" / "figures")
    p.add_argument("--hr-csv", type=Path, default=DEFAULT_HR_CSV)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _load_cohort(args.train, args.test)

    plot_km_overall(d, out_dir / "01_km_overall.png")
    plot_km_by_column(d, "regimen", out_dir / "02_km_by_regimen.png", "Kaplan–Meier by regimen")
    plot_km_by_column(d, "apoe_e4", out_dir / "03_km_by_apoe.png", "Kaplan–Meier by APOE ε4 carrier status")
    plot_nelson_aalen(d, out_dir / "04_nelson_aalen_overall.png")
    plot_time_distributions(d, out_dir / "05_time_diagnostics.png")
    plot_regimen_counts(d, out_dir / "06_regimen_counts.png")
    plot_eta_vs_texact(d, out_dir / "07_eta_vs_T_exact.png")
    plot_covariate_marginals(d, out_dir / "08_covariate_marginals.png")
    plot_binary_prevalence(d, out_dir / "09_binary_prevalence.png")
    plot_target_comparison(d, out_dir / "10_target_event_rates.png")
    wrote_forest = plot_hr_recovery_forest(args.hr_csv, out_dir / "11_hr_recovery_forest.png")
    plot_km_train_test(d, out_dir / "12_km_train_vs_test.png")

    summary_path = out_dir / "summary_metrics.txt"
    write_summary_metrics(d, summary_path)

    print(f"[visualize_cici_survival_cohort] Wrote figures to: {out_dir}")
    print(f"[visualize_cici_survival_cohort] Wrote: {summary_path}")
    if not wrote_forest:
        print(f"[visualize_cici_survival_cohort] Skipped HR forest (missing or empty {args.hr_csv})")


if __name__ == "__main__":
    main()
