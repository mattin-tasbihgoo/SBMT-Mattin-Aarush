"""D1: score the 168 pre-registered inference/post-processing configurations and choose by cross-fitting.

See the D1 pre-registration in results/RUNS.md and PROTOCOL.md section 5.1. For fold k, the
configuration with the highest mean Dice over the OOF subjects of the other four folds is
chosen (ties: first in pre-registered order) and applied to fold k. The cross-fitted result
is compared with E0 (I0, threshold 0.5, largest CC) by the ship rule.

    python src/evaluation/d1_crossfit.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.error_analysis_3d import case_metrics, RAW
from evaluation.cv_report import summarize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
E0_PROBS = PROJECT_ROOT / "outputs" / "cv_e0_v1recipe_2026-11-15" / "oof_probs"
D1_PROBS = PROJECT_ROOT / "outputs" / "d1_inference_2026-11-16" / "probs"
RES = PROJECT_ROOT / "results" / "d1_2026-11-16"
VARIANTS = ("I0", "I1", "I2", "I3", "I4", "I5")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
POSTS = ("none", "lcc", "min100", "min1000", "min5000", "rel10", "rel25")
BASELINE = ("I0", 0.5, "lcc")
E0_BOTTOM_DECILE = 0.7242
N_BOOT, BOOT_SEED = 10_000, 20261115
EPS = 1e-6


def load_prob(variant, sid):
    path = E0_PROBS / f"{sid}.npy" if variant == "I0" else D1_PROBS / variant / f"{sid}.npy"
    return np.load(path).astype(np.float32)


def keep_mask(sizes, post):
    """sizes: component sizes (index 0 = component 1). Returns boolean keep vector."""
    if len(sizes) == 0 or post == "none":
        return np.ones(len(sizes), bool)
    if post == "lcc":
        k = np.zeros(len(sizes), bool)
        k[sizes.argmax()] = True  # first largest, as largest_connected_component()
        return k
    if post.startswith("min"):
        return sizes >= int(post[3:])
    return sizes >= sizes.max() * int(post[3:]) / 100.0


def subject_scores(sid):
    gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
    g = int(gt.sum())
    rows = []
    for v in VARIANTS:
        prob = load_prob(v, sid)
        for t in THRESHOLDS:
            lab, n = ndi.label(prob >= t)
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            inter = np.bincount(lab[gt], minlength=n + 1)[1:]
            for post in POSTS:
                k = keep_mask(sizes, post)
                d = (2.0 * inter[k].sum() + EPS) / (sizes[k].sum() + g + EPS)
                rows.append((sid, v, t, post, float(d)))
    return rows


def final_rows(args):
    sid, (v, t, post) = args
    import nibabel as nib
    gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
    labv = np.asarray(nib.load(str(RAW / "labelsTr" / f"{sid}.nii.gz")).dataobj).astype(np.uint8)
    lab, n = ndi.label(load_prob(v, sid) >= t)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = np.concatenate([[False], keep_mask(sizes, post)])
    pred = keep[lab]
    return {"subject_id": sid, "variant": v, "threshold": t, "post": post, **case_metrics(pred, gt, labv)}


def main():
    if (RES / "summary.csv").exists():
        sys.exit(f"{RES / 'summary.csv'} exists; refusing to overwrite.")
    RES.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(FOLDS_CSV)
    sids = list(folds.subject_id)
    with ProcessPoolExecutor(max_workers=20) as ex:
        out = list(ex.map(subject_scores, sids, chunksize=2))
    grid = pd.DataFrame([r for rows in out for r in rows],
                        columns=["subject_id", "variant", "threshold", "post", "dice"])
    grid = grid.merge(folds[["subject_id", "fold"]], on="subject_id")
    grid.to_csv(RES / "grid_dice_long.csv.gz", index=False, float_format="%.6f")

    # Self-check: the baseline config reproduces E0's per-case Dice.
    e0 = pd.read_csv(PROJECT_ROOT / "results" / "cv_e0_2026-11-15" / "per_case_oof.csv")
    e0 = e0[e0.post == "cc"][["subject_id", "dice"]]
    b = grid[(grid.variant == BASELINE[0]) & (grid.threshold == BASELINE[1]) & (grid.post == BASELINE[2])]
    chk = b.merge(e0, on="subject_id", suffixes=("", "_e0"))
    print(f"baseline vs E0 per-case Dice: max |diff| = {(chk.dice - chk.dice_e0).abs().max():.1e}")

    order = [(v, t, p) for v in VARIANTS for t in THRESHOLDS for p in POSTS]
    rank = {c: i for i, c in enumerate(order)}
    keys = ["variant", "threshold", "post"]

    # Cross-fitting: choose on the other four folds, apply to fold k.
    choices, picked = [], []
    for k in range(5):
        m = grid[grid.fold != k].groupby(keys).dice.mean().reset_index()
        m["rank"] = [rank[tuple(r)] for r in m[keys].itertuples(index=False)]
        m = m.sort_values(["dice", "rank"], ascending=[False, True])
        best = tuple(m.iloc[0][keys])
        base_mean = m[(m.variant == BASELINE[0]) & (m.threshold == BASELINE[1]) & (m.post == BASELINE[2])].dice.iloc[0]
        choices.append({"fold": k, "variant": best[0], "threshold": best[1], "post": best[2],
                        "selection_mean_dice_other_folds": m.iloc[0].dice,
                        "baseline_mean_dice_other_folds": base_mean,
                        "runner_up": " ".join(map(str, m.iloc[1][keys])), "runner_up_dice": m.iloc[1].dice})
        sel = grid[(grid.fold == k) & (grid.variant == best[0]) & (grid.threshold == best[1]) & (grid.post == best[2])]
        picked.append(sel)
    choices = pd.DataFrame(choices)
    choices.to_csv(RES / "crossfit_choices.csv", index=False, float_format="%.6f")
    print(choices.to_string(index=False))

    cf = pd.concat(picked)[["subject_id", "fold", "variant", "threshold", "post", "dice"]]
    both = cf.merge(b[["subject_id", "dice"]], on="subject_id", suffixes=("", "_e0"))
    diff = (both.dice - both.dice_e0).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    boots = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bottom = np.sort(both.dice.to_numpy())[:48].mean()
    ships = bool(lo > 0 and bottom >= E0_BOTTOM_DECILE)
    print(f"cross-fitted mean {both.dice.mean():.4f} vs E0 {both.dice_e0.mean():.4f}; "
          f"paired diff {diff.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); "
          f"bottom decile {bottom:.4f} vs {E0_BOTTOM_DECILE}; better/worse >0.005: "
          f"{(diff > 0.005).sum()}/{(diff < -0.005).sum()}; SHIPS: {ships}")

    # Full per-case metrics for the cross-fitted result.
    jobs = [(r.subject_id, (r.variant, r.threshold, r.post)) for r in cf.itertuples()]
    with ProcessPoolExecutor(max_workers=20) as ex:
        full = pd.DataFrame(list(ex.map(final_rows, jobs, chunksize=4)))
    full = full.merge(folds[["subject_id", "fold", "april_val96"]], on="subject_id")
    full = full.merge(b[["subject_id", "dice"]].rename(columns={"dice": "dice_e0"}), on="subject_id")
    full.to_csv(RES / "per_case_crossfit.csv", index=False, float_format="%.6f")

    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for name, s in [("all_484", full), ("april_val96", full[full.april_val96 == 1])] + \
                   [(f"fold{k}", full[full.fold == k]) for k in range(5)]:
        for metric in ["dice", "sdice_1mm", "sdice_2mm", "hd95", "assd"]:
            rows.append({"subset": name, "metric": metric, **summarize(s[metric], rng)})
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "summary.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"mean_diff": diff.mean(), "ci_lo": lo, "ci_hi": hi, "bottom_decile": bottom,
                   "e0_bottom_decile": E0_BOTTOM_DECILE, "n_better_0.005": int((diff > 0.005).sum()),
                   "n_worse_0.005": int((diff < -0.005).sum()), "ships": ships}]
                 ).to_csv(RES / "ship_rule.csv", index=False, float_format="%.4f")

    # Descriptive only (NOT a reportable number): pooled ranking of all 168 configs on 484.
    pooled = grid.groupby(keys).dice.agg(["mean", "median"]).reset_index().sort_values("mean", ascending=False)
    pooled.to_csv(RES / "pooled_config_means_descriptive.csv", index=False, float_format="%.5f")
    print(summ[summ.subset.isin(["all_484", "april_val96"]) & (summ.metric != "assd")]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("top 10 pooled (descriptive):")
    print(pooled.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
