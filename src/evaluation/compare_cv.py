"""Aggregate a step-E CV run and apply the ship rule against the current baseline (E0 + D1).

Frozen inference/post-processing = D1 choice (I2, threshold 0.5, keep components >= 10% of
the largest). Writes per-case metrics (same definitions as error_analysis_3d.py), the
protocol summary and the ship-rule result to results/<tag>/. Refuses to overwrite.

    python src/evaluation/compare_cv.py --run-dir outputs/cv_e1_4ch_2026-11-17 --tag cv_e1_2026-11-17
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.error_analysis_3d import case_metrics, RAW
from evaluation.cv_report import summarize
from evaluation.d1_crossfit import keep_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
BASELINE_CSV = PROJECT_ROOT / "results" / "d1_2026-11-16" / "per_case_crossfit.csv"
BASELINE_BOTTOM_DECILE = 0.7542
N_BOOT, BOOT_SEED = 10_000, 20261115


def subject_row(sid, prob_dir):
    import nibabel as nib
    gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
    labv = np.asarray(nib.load(str(RAW / "labelsTr" / f"{sid}.nii.gz")).dataobj).astype(np.uint8)
    lab, n = ndi.label(np.load(prob_dir / f"{sid}.npy").astype(np.float32) >= 0.5)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    pred = np.concatenate([[False], keep_mask(sizes, "rel10")])[lab]
    return {"subject_id": sid, **case_metrics(pred, gt, labv)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    res = PROJECT_ROOT / "results" / args.tag
    if (res / "summary.csv").exists():
        sys.exit(f"{res / 'summary.csv'} exists; refusing to overwrite.")
    res.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(FOLDS_CSV)
    quick = pd.concat([pd.read_csv(run_dir / f"fold{k}" / "test_per_case.csv") for k in range(5)])
    assert sorted(quick.subject_id) == sorted(folds.subject_id)

    with ProcessPoolExecutor(max_workers=20) as ex:
        df = pd.DataFrame(list(ex.map(partial(subject_row, prob_dir=run_dir / "oof_probs" / "I2"),
                                      folds.subject_id, chunksize=4)))
    df = df.merge(folds[["subject_id", "fold", "april_val96"]], on="subject_id")
    df = df.merge(quick[["subject_id", "dice_d1", "dice_e0cfg"]], on="subject_id")
    print(f"max |Dice(float16 maps) - Dice(score_fold_e)| = {(df.dice - df.dice_d1).abs().max():.1e}")
    base = pd.read_csv(BASELINE_CSV)[["subject_id", "dice", "sdice_2mm", "hd95",
                                      "fn_edema_vox", "fn_nonenh_vox", "fn_enh_vox"]]
    df = df.merge(base.add_suffix("_base").rename(columns={"subject_id_base": "subject_id"}), on="subject_id")
    df.to_csv(res / "per_case_oof.csv", index=False, float_format="%.6f")

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for name, s in [("all_484", df), ("april_val96", df[df.april_val96 == 1])] + \
                   [(f"fold{k}", df[df.fold == k]) for k in range(5)]:
        for m in ["dice", "sdice_1mm", "sdice_2mm", "hd95", "assd"]:
            rows.append({"subset": name, "metric": m, **summarize(s[m], rng)})
    summ = pd.DataFrame(rows)
    summ.to_csv(res / "summary.csv", index=False, float_format="%.4f")

    diff = (df.dice - df.dice_base).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    boots = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bottom = np.sort(df.dice.to_numpy())[:48].mean()
    ships = bool(lo > 0 and bottom >= BASELINE_BOTTOM_DECILE)
    rule = {"mean": df.dice.mean(), "baseline_mean": df.dice_base.mean(), "mean_diff": diff.mean(),
            "ci_lo": lo, "ci_hi": hi, "bottom_decile": bottom, "baseline_bottom_decile": BASELINE_BOTTOM_DECILE,
            "n_better_0.005": int((diff > 0.005).sum()), "n_worse_0.005": int((diff < -0.005).sum()),
            "ships": ships}
    pd.DataFrame([rule]).to_csv(res / "ship_rule.csv", index=False, float_format="%.4f")
    for k, v in rule.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    miss = {n: (df[f"fn_{n}_vox"].sum() / df[f"gt_{n}_vox"].sum(),
                df[f"fn_{n}_vox_base"].sum() / df[f"gt_{n}_vox"].sum()) for n in ("edema", "nonenh", "enh")}
    print("missed-voxel share (variant, baseline):", {k: tuple(round(x, 4) for x in v) for k, v in miss.items()})
    print("E0-config mean (descriptive):", round(df.dice_e0cfg.mean(), 4))
    print(summ[summ.subset.isin(["all_484", "april_val96"]) & (summ.metric == "dice")]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
