"""Audit of the April 2026 3D U-Net evaluation (handoff section 6).

Reads files only (no GPU, no training). Run from the repo root after the four
evaluate_3d.py runs have written outputs/<AUDIT_TAG>/{v1,v2,ensemble,v2_latest}:

  python src/evaluation/audit_3d.py

Writes results/<AUDIT_TAG>/:
  summary.csv                    distribution summary per model (section 3 metrics)
  april_vs_regenerated.csv       per-case April Dice vs regenerated Dice
  {v1,v2,ensemble,v2_latest}_subject_metrics.csv, *_run_config.json   copies
and prints markdown used in results/AUDIT.md.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_TAG = "audit_2026-11-15"
OUT = PROJECT_ROOT / "outputs"
EVAL = OUT / AUDIT_TAG
RES = PROJECT_ROOT / "results" / AUDIT_TAG

RUNS = ["v1", "v2", "ensemble", "v2_latest"]
APRIL = {
    "v1": OUT / "3d_unet_v1" / "evaluation" / "subject_metrics_3d.csv",
    "v2": OUT / "3d_unet_v2" / "evaluation" / "subject_metrics_3d.csv",
    "ensemble": OUT / "3d_unet_ensemble_v1v2" / "evaluation" / "subject_metrics_3d.csv",
}
SPLIT_DIRS = {
    "reference (2.5D baseline)": OUT / "runtime_flair_t1gd_25d5_smalllesion_baseline_v1" / "logs",
    "3d_unet_v1": OUT / "3d_unet_v1" / "logs",
    "3d_unet_v2": OUT / "3d_unet_v2" / "logs",
}
TRAIN_LOGS = {
    "v1": OUT / "3d_unet_v1" / "logs" / "training_log.csv",
    "v2": OUT / "3d_unet_v2" / "logs" / "training_log.csv",
}
N_BOOT = 10_000
BOOT_SEED = 20261115


def read_list(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(x: np.ndarray, rng) -> dict:
    lo, hi = boot_ci(x, rng)
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": len(x), "mean": x.mean(), "ci95_lo": lo, "ci95_hi": hi,
        "median": np.median(x), "q1": q1, "q3": q3, "iqr": q3 - q1,
        "sd_ddof1": x.std(ddof=1), "sd_ddof0": x.std(ddof=0), "min": x.min(),
        "frac_ge_0.80": (x >= 0.80).mean(), "frac_ge_0.90": (x >= 0.90).mean(),
    }


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_check():
    print("## Split check\n")
    raw_ids = sorted(p.name.replace(".nii.gz", "") for p in
                     (PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour" / "labelsTr").glob("BRATS_*.nii.gz"))
    proc_ids = sorted(p.stem for p in (PROJECT_ROOT / "data" / "processed_3d" / "images").glob("*.npy"))
    rows = []
    lists = {}
    for name, d in SPLIT_DIRS.items():
        tr, va = read_list(d / "train_subjects.txt"), read_list(d / "val_subjects.txt")
        lists[name] = (tr, va)
        rows.append({
            "source": name, "train": len(tr), "train_unique": len(set(tr)),
            "val": len(va), "val_unique": len(set(va)),
            "overlap": len(set(tr) & set(va)), "union": len(set(tr) | set(va)),
            "union_eq_processed_484": set(tr) | set(va) == set(proc_ids),
            "train_sha256": sha256(d / "train_subjects.txt")[:12],
            "val_sha256": sha256(d / "val_subjects.txt")[:12],
        })
    df = pd.DataFrame(rows)
    print(md_table(df))
    ref = list(lists.values())[0]
    same_order = all(v == ref for v in lists.values())
    print(f"\nraw labelsTr subjects: {len(raw_ids)}; processed_3d subjects: {len(proc_ids)}; "
          f"identical sets: {raw_ids == proc_ids}")
    print(f"all three split copies identical (same lists, same order): {same_order}\n")
    return ref[1]


def checkpoint_selection(val, regen):
    """Which validation subjects chose the checkpoint? Recompute the logged sw_val_dice."""
    print("## Checkpoint selection: which subjects chose the best epoch\n")
    rows = []
    for run, version, epoch_rule in [("v1", "v1", "best"), ("v2", "v2", "best"), ("v2_latest", "v2", "last")]:
        log = pd.read_csv(TRAIN_LOGS[version])
        if epoch_rule == "best":
            r = log.loc[log["sw_val_dice"].idxmax()]
        else:
            r = log.iloc[-1]
        df = regen[run].set_index("subject_id")
        for n in (5, 10):
            subs = val[:n]
            m = df.loc[subs, "dice_nocc"].mean()
            rows.append({"run": run, "epoch": int(r["epoch"]), "logged_sw_val_dice": float(r["sw_val_dice"]),
                         "first_n_val": n, "recomputed_mean_nocc": float(m),
                         "abs_diff": abs(float(m) - float(r["sw_val_dice"]))})
    df = pd.DataFrame(rows)
    print(md_table(df, "{:.6f}"))
    print(f"\nfirst 10 val subjects (file order): {', '.join(val[:10])}\n")
    return df


def poster_panel(regen):
    print("## Poster panels and overlay folders\n")
    v1 = regen["v1"].set_index("subject_id")
    for sid, printed in [("BRATS_178", 0.7436), ("BRATS_051", 0.9644), ("BRATS_381", 0.555)]:
        print(f"{sid}: printed {printed}  |  V1 regenerated dice_nocc={v1.loc[sid, 'dice_nocc']:.4f}  "
              f"dice_3d(CC)={v1.loc[sid, 'dice_3d']:.4f}")
    print()
    for version in ("v1", "v2"):
        df = regen[version]
        for metric in ("dice_nocc", "dice_3d"):
            s = df.sort_values(metric, ascending=False)
            best, worst = set(s.head(10).subject_id), set(s.tail(10).subject_id)
            vdir = OUT / f"3d_unet_{version}" / "visualizations" / "overlays"
            fb = {p.stem for p in (vdir / "best").glob("*.png")}
            fw = {p.stem for p in (vdir / "worst").glob("*.png")}
            print(f"{version} overlays vs top/bottom-10 by {metric}: best match {len(best & fb)}/10, "
                  f"worst match {len(worst & fw)}/10")
    s = regen["v1"].sort_values("dice_nocc")
    print("\nV1 lowest 10 by no-CC Dice:")
    print(md_table(s.head(10)[["subject_id", "tumor_voxels", "dice_nocc", "dice_3d", "n_components"]]))
    print()


def main():
    RES.mkdir(parents=True, exist_ok=True)
    regen = {r: pd.read_csv(EVAL / r / "subject_metrics_3d.csv") for r in RUNS}
    for r in RUNS:
        shutil.copy2(EVAL / r / "subject_metrics_3d.csv", RES / f"{r}_subject_metrics.csv")
        shutil.copy2(EVAL / r / "run_config.json", RES / f"{r}_run_config.json")

    val = split_check()
    for r in RUNS:
        assert set(regen[r].subject_id) == set(val) and len(regen[r]) == 96, r

    # ── April vs regenerated ─────────────────────────────────────────
    print("## April vs regenerated (thr 0.5 + largest CC)\n")
    rows, cmp_frames = [], []
    for r, p in APRIL.items():
        a = pd.read_csv(p)[["subject_id", "dice_3d"]].rename(columns={"dice_3d": "april"})
        g = regen[r][["subject_id", "dice_3d"]].rename(columns={"dice_3d": "regen"})
        m = a.merge(g, on="subject_id", validate="1:1")
        m["abs_diff"] = (m.regen - m.april).abs()
        m.insert(0, "model", r)
        cmp_frames.append(m)
        rows.append({"model": r, "april_mean": m.april.mean(), "regen_mean": m.regen.mean(),
                     "april_median": m.april.median(), "regen_median": m.regen.median(),
                     "max_case_abs_diff": m.abs_diff.max(),
                     "cases_diff_gt_1e-3": int((m.abs_diff > 1e-3).sum())})
    print(md_table(pd.DataFrame(rows)))
    pd.concat(cmp_frames).to_csv(RES / "april_vs_regenerated.csv", index=False)
    print()

    # ── Hypothesis: April V1/V2 single-model CSVs were produced with flip TTA ──
    tta_runs = [r for r in ("v1_tta", "v2_tta") if (EVAL / r / "subject_metrics_3d.csv").exists()]
    if tta_runs:
        print("## April single-model CSVs vs regenerated *with flip TTA* (thr 0.5 + largest CC)\n")
        trows = []
        for r in tta_runs:
            v = r.split("_")[0]
            shutil.copy2(EVAL / r / "subject_metrics_3d.csv", RES / f"{r}_subject_metrics.csv")
            shutil.copy2(EVAL / r / "run_config.json", RES / f"{r}_run_config.json")
            a = pd.read_csv(APRIL[v]).set_index("subject_id")["dice_3d"]
            g = pd.read_csv(EVAL / r / "subject_metrics_3d.csv").set_index("subject_id")
            d = (g["dice_3d"] - a).abs()
            at = pd.read_csv(APRIL[v].parent / "threshold_sweep_3d.csv").set_index("threshold")["mean_dice"]
            gt = pd.read_csv(EVAL / r / "threshold_sweep_3d.csv").set_index("threshold")["mean_dice"]
            trows.append({"model": r, "april_mean": a.mean(), "regen_mean": g["dice_3d"].mean(),
                          "april_median": a.median(), "regen_median": g["dice_3d"].median(),
                          "max_case_abs_diff": d.max(), "cases_diff_gt_1e-3": int((d > 1e-3).sum()),
                          "max_sweep_mean_diff": (gt - at).abs().max()})
        print(md_table(pd.DataFrame(trows)))
        print()

    # ── Distribution summary ─────────────────────────────────────────
    print("## Distribution (regenerated 2026-11-15; thr 0.5)\n")
    rng = np.random.default_rng(BOOT_SEED)
    srows = []
    for r in RUNS:
        for metric, label in [("dice_3d", "largest CC"), ("dice_nocc", "no CC")]:
            s = summarize(regen[r][metric].to_numpy(), rng)
            srows.append({"model": r, "postproc": label, **s})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(RES / "summary.csv", index=False)
    show = sdf[sdf.postproc == "largest CC"].drop(columns=["postproc", "q1", "q3"])
    print(md_table(show))
    print("\nno-CC variant:")
    print(md_table(sdf[sdf.postproc == "no CC"][["model", "mean", "ci95_lo", "ci95_hi", "median", "min"]]))
    print()

    # ── Paired differences ───────────────────────────────────────────
    print("## Paired differences in mean Dice (bootstrap over subjects)\n")
    prow = []
    base = {r: regen[r].set_index("subject_id").loc[val, "dice_3d"].to_numpy() for r in RUNS}
    base_nocc = {r: regen[r].set_index("subject_id").loc[val, "dice_nocc"].to_numpy() for r in RUNS}
    pairs = [("ensemble", "v1", base), ("ensemble", "v2", base), ("v1", "v2", base),
             ("v2", "v2_latest", base)]
    for a, b, src in pairs:
        d = src[a] - src[b]
        lo, hi = boot_ci(d, rng)
        prow.append({"comparison": f"{a} - {b} (CC)", "mean_diff": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
                     "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum())})
    for r in ("v1", "v2", "ensemble"):
        d = base[r] - base_nocc[r]
        lo, hi = boot_ci(d, rng)
        prow.append({"comparison": f"{r}: CC - noCC", "mean_diff": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
                     "a_better": int((d > 1e-9).sum()), "b_better": int((d < -1e-9).sum())})
    print(md_table(pd.DataFrame(prow)))
    print()

    checkpoint_selection(val, regen)
    poster_panel(regen)

    # ── Threshold sweep (regenerated) ────────────────────────────────
    print("## Threshold sweep, regenerated (mean Dice, largest CC)\n")
    trows = []
    for r in RUNS:
        t = pd.read_csv(EVAL / r / "threshold_sweep_3d.csv")
        trows.append({"model": r, **{f"t{x:.1f}": y for x, y in zip(t.threshold, t.mean_dice)}})
    print(md_table(pd.DataFrame(trows)))
    print()

    # ── Lowest cases ─────────────────────────────────────────────────
    print("## Lowest 8 cases, ensemble (with V1/V2 columns)\n")
    e = regen["ensemble"][["subject_id", "tumor_voxels", "dice_3d", "dice_nocc", "n_components"]]
    e = e.merge(regen["v1"][["subject_id", "dice_3d"]].rename(columns={"dice_3d": "v1"}), on="subject_id")
    e = e.merge(regen["v2"][["subject_id", "dice_3d"]].rename(columns={"dice_3d": "v2"}), on="subject_id")
    print(md_table(e.sort_values("dice_3d").head(8)))
    print(f"\nWrote {RES}")


if __name__ == "__main__":
    main()
