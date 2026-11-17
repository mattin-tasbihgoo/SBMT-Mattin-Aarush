"""Aggregate a finished CV run into out-of-fold (OOF) results (results/PROTOCOL.md).

Reads every fold's saved float16 probability maps, applies the frozen
baseline post-processing (threshold 0.5 + largest CC; also reports no-CC),
computes per-case Dice, surface Dice @1/2 mm, HD95, ASSD and error volumes
(same definitions as error_analysis_3d.py), and writes the per-case CSV and
the protocol summary (mean, median, IQR, SD, min, frac >= 0.80/0.90,
10k-bootstrap CI) to results/<tag>/.

    python src/evaluation/cv_report.py --run-dir outputs/cv_e0_v1recipe_2026-11-15 --tag cv_e0_2026-11-15
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.error_analysis_3d import case_metrics, largest_cc, RAW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
THR = 0.5
N_BOOT = 10_000
BOOT_SEED = 20261115
METRICS = ["dice", "sdice_1mm", "sdice_2mm", "hd95", "assd"]


def subject_rows(sid, prob_dir):
    import nibabel as nib
    gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
    lab = np.asarray(nib.load(str(RAW / "labelsTr" / f"{sid}.nii.gz")).dataobj).astype(np.uint8)
    assert np.array_equal(gt, lab > 0), sid
    prob = np.load(prob_dir / f"{sid}.npy").astype(np.float32)
    raw = prob >= THR
    rows = []
    for post, pred in (("nocc", raw), ("cc", largest_cc(raw))):
        rows.append({"subject_id": sid, "post": post, **case_metrics(pred, gt, lab)})
    return rows


def summarize(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    boots = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": len(x), "mean": x.mean(), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "median": np.median(x), "q1": q1, "q3": q3,
            "sd": x.std(ddof=1), "min": x.min(), "max": x.max(),
            "frac_ge_0.80": (x >= 0.80).mean(), "frac_ge_0.90": (x >= 0.90).mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    res = PROJECT_ROOT / "results" / args.tag
    if (res / "summary.csv").exists():
        sys.exit(f"{res / 'summary.csv'} exists; refusing to overwrite.")
    res.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(FOLDS_CSV)
    fold_csvs = [run_dir / f"fold{k}" / "test_per_case.csv" for k in range(5)]
    missing = [str(c) for c in fold_csvs if not c.exists()]
    if missing:
        sys.exit("folds not scored yet:\n" + "\n".join(missing))
    quick = pd.concat([pd.read_csv(c) for c in fold_csvs])
    assert sorted(quick.subject_id) == sorted(folds.subject_id), "OOF set != all 484 subjects"

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(partial(subject_rows, prob_dir=run_dir / "oof_probs"),
                          folds.subject_id, chunksize=4))
    df = pd.DataFrame([r for rows in out for r in rows])
    df = df.merge(folds[["subject_id", "fold", "april_val96"]], on="subject_id")
    # Self-check: Dice recomputed from the float16 maps vs Dice from evaluate_cv.py (float32 maps).
    chk = df[df.post == "cc"].merge(quick[["subject_id", "dice_cc"]], on="subject_id")
    dmax = (chk.dice - chk.dice_cc).abs().max()
    print(f"max |Dice(float16 maps) - Dice(evaluate_cv)| = {dmax:.2e}")
    df.to_csv(res / "per_case_oof.csv", index=False, float_format="%.6f")

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for post in ("cc", "nocc"):
        d = df[df.post == post]
        subsets = [("all_484", d), ("april_val96", d[d.april_val96 == 1])]
        subsets += [(f"fold{k}", d[d.fold == k]) for k in range(5)]
        for name, s in subsets:
            for m in METRICS:
                rows.append({"post": post, "subset": name, "metric": m, **summarize(s[m], rng)})
    summ = pd.DataFrame(rows)
    summ.to_csv(res / "summary.csv", index=False, float_format="%.4f")
    show = summ[(summ.metric == "dice") & summ.subset.isin(["all_484", "april_val96"])]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
