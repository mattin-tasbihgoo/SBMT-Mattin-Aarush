"""Step B error analysis of the April 2026 3D U-Net (V1, V2, ensemble) on the
96 frozen validation subjects. Analysis only: nothing here selects a
configuration (the 96 already made the April selection decisions).

Stages (run from the repo root, in order):
  python src/evaluation/error_analysis_3d.py infer     # GPU: save V1/V2 probability maps (float32)
  python src/evaluation/error_analysis_3d.py metrics   # CPU (multiprocess): per-case metrics
  python src/evaluation/error_analysis_3d.py plots     # summary plots (no patient images)
  python src/evaluation/error_analysis_3d.py figures   # per-case review panels (MRI; kept in outputs/)

Inference is the headline April configuration (128^3 sliding window, 50 %
overlap, count-map blending, AMP, no TTA); thresholding at 0.5 and optional
largest-connected-component (6-connectivity, as in evaluate_3d.py) are applied
afterwards, so every post-processing variant comes from the same probabilities.
The metrics stage asserts that Dice reproduces results/audit_2026-11-15 exactly.

Metric definitions (voxel spacing 1 mm isotropic, from the NIfTI headers):
  surface voxels   mask & ~erode(mask, 6-connectivity), outside volume = background
  surface Dice @t  (#pred-surface voxels within t mm of GT surface
                    + #GT-surface voxels within t mm of pred surface) / (#pred-surf + #GT-surf)
                   (voxel-count approximation of Nikolov et al. 2021, not mesh-area weighted)
  HD95             max of the two directed 95th-percentile surface distances (MONAI convention)
  ASSD             mean of all surface-to-surface distances, both directions
  boundary FN/FP   FN voxels within 2 mm of the predicted mask / FP voxels within 2 mm of GT;
                   the rest are "gross" (missed or spurious regions)
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TAG = "error_analysis_2026-11-15"
OUT = PROJECT_ROOT / "outputs" / TAG
RES = PROJECT_ROOT / "results" / TAG
PROB_DIR = OUT / "probs"
FIG_DIR = OUT / "case_figures"
DATA = PROJECT_ROOT / "data" / "processed_3d"
RAW = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
VAL_TXT = PROJECT_ROOT / "outputs" / "runtime_flair_t1gd_25d5_smalllesion_baseline_v1" / "logs" / "val_subjects.txt"
AUDIT_DIR = PROJECT_ROOT / "results" / "audit_2026-11-15"

SPACING = (1.0, 1.0, 1.0)
THR = 0.5
TOLS = (1.0, 2.0)
BOUNDARY_MM = 2.0
MODELS = ("v1", "v2", "ensemble")
LABEL_NAMES = {1: "edema", 2: "nonenh", 3: "enh"}
N_WORKERS = 20
N_BOOT = 10_000
BOOT_SEED = 20261115
STRUCT6 = ndi.generate_binary_structure(3, 1)


def val_subjects() -> list[str]:
    return [l.strip() for l in VAL_TXT.read_text().splitlines() if l.strip()]


# ── Stage 1: inference ──────────────────────────────────────────────────

def stage_infer():
    import torch
    from evaluation.evaluate_3d import CHECKPOINTS, build_model, set_seed, PATCH_SIZE, OVERLAP, RANDOM_SEED
    from evaluation.sliding_window import sliding_window_inference
    from tqdm import tqdm

    set_seed(RANDOM_SEED)
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA torch required for the infer stage"
    for version in ("v1", "v2"):
        d = PROB_DIR / version
        d.mkdir(parents=True, exist_ok=True)
        model = build_model(version, device)
        model.load_state_dict(torch.load(CHECKPOINTS[version], map_location=device, weights_only=True), strict=True)
        model.eval()
        for sid in tqdm(val_subjects(), desc=f"infer {version}"):
            out = d / f"{sid}.npy"
            if out.exists():
                continue
            image = np.load(DATA / "images" / f"{sid}.npy")
            prob = sliding_window_inference(model, image, PATCH_SIZE, OVERLAP, device, True)
            np.save(out, prob.astype(np.float32))
        del model
        torch.cuda.empty_cache()


# ── Stage 2: metrics ────────────────────────────────────────────────────

def largest_cc(binary):
    lab, n = ndi.label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    return float((2.0 * inter + 1e-6) / (a.sum() + b.sum() + 1e-6))


def surface(mask):
    return mask & ~ndi.binary_erosion(mask, structure=STRUCT6, border_value=0)


def crop_box(*masks, pad=3):
    union = np.logical_or.reduce(masks)
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, union.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def case_metrics(pred, gt, lab):
    tp = pred & gt
    fp = pred & ~gt
    fn = gt & ~pred
    r = {"dice": dice(pred, gt), "gt_vox": int(gt.sum()), "pred_vox": int(pred.sum()),
         "tp_vox": int(tp.sum()), "fp_vox": int(fp.sum()), "fn_vox": int(fn.sum()),
         "pred_components": int(ndi.label(pred)[1])}
    for k, name in LABEL_NAMES.items():
        gk = lab == k
        r[f"gt_{name}_vox"] = int(gk.sum())
        r[f"fn_{name}_vox"] = int((fn & gk).sum())
    if pred.sum() == 0:
        r.update({f"sdice_{t:g}mm": 0.0 for t in TOLS})
        r.update({"hd95": np.nan, "assd": np.nan, "hd_max": np.nan,
                  "fn_boundary_vox": 0, "fn_gross_vox": r["fn_vox"], "fp_boundary_vox": 0, "fp_gross_vox": 0})
        return r
    box = crop_box(pred, gt)
    p, g = pred[box], gt[box]
    sp, sg = surface(p), surface(g)
    dt_sg = ndi.distance_transform_edt(~sg, sampling=SPACING)
    dt_sp = ndi.distance_transform_edt(~sp, sampling=SPACING)
    d_p2g, d_g2p = dt_sg[sp], dt_sp[sg]
    n = len(d_p2g) + len(d_g2p)
    for t in TOLS:
        r[f"sdice_{t:g}mm"] = float(((d_p2g <= t).sum() + (d_g2p <= t).sum()) / n)
    r["hd95"] = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    r["assd"] = float((d_p2g.sum() + d_g2p.sum()) / n)
    r["hd_max"] = float(max(d_p2g.max(), d_g2p.max()))
    dist_to_pred = ndi.distance_transform_edt(~p, sampling=SPACING)
    dist_to_gt = ndi.distance_transform_edt(~g, sampling=SPACING)
    fnb, fpb = fn[box], fp[box]
    r["fn_boundary_vox"] = int((fnb & (dist_to_pred <= BOUNDARY_MM)).sum())
    r["fn_gross_vox"] = int(fnb.sum()) - r["fn_boundary_vox"]
    r["fp_boundary_vox"] = int((fpb & (dist_to_gt <= BOUNDARY_MM)).sum())
    r["fp_gross_vox"] = int(fpb.sum()) - r["fp_boundary_vox"]
    return r


def subject_rows(sid):
    gt = np.load(DATA / "masks" / f"{sid}.npy").astype(bool)
    import nibabel as nib
    lab = np.asarray(nib.load(str(RAW / "labelsTr" / f"{sid}.nii.gz")).dataobj).astype(np.uint8)
    assert np.array_equal(gt, lab > 0), sid
    p1 = np.load(PROB_DIR / "v1" / f"{sid}.npy")
    p2 = np.load(PROB_DIR / "v2" / f"{sid}.npy")
    acc = np.zeros_like(p1)
    acc += p1
    acc += p2
    probs = {"v1": p1, "v2": p2, "ensemble": acc / 2}  # same arithmetic as evaluate_3d.py

    gl, gn = ndi.label(gt)
    gsizes = np.bincount(gl.ravel())[1:]
    base = {"subject_id": sid, "gt_components": int(gn),
            "gt_frac_largest_component": float(gsizes.max() / gsizes.sum()) if gn else np.nan}
    rows = []
    for m in MODELS:
        raw = probs[m] >= THR
        cc = largest_cc(raw)
        removed = raw & ~cc
        for post, pred in (("nocc", raw), ("cc", cc)):
            r = {**base, "model": m, "post": post, **case_metrics(pred, gt, lab)}
            if post == "cc":
                r["cc_removed_tp_vox"] = int((removed & gt).sum())
                r["cc_removed_fp_vox"] = int((removed & ~gt).sum())
            rows.append(r)
    return rows


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def stage_metrics():
    RES.mkdir(parents=True, exist_ok=True)
    subs = val_subjects()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = [r for rs in ex.map(subject_rows, subs, chunksize=2) for r in rs]
    df = pd.DataFrame(rows).sort_values(["model", "post", "subject_id"]).reset_index(drop=True)

    # Self-check against the audit: Dice must match exactly.
    for m in MODELS:
        a = pd.read_csv(AUDIT_DIR / f"{m}_subject_metrics.csv").set_index("subject_id")
        for post, col in (("cc", "dice_3d"), ("nocc", "dice_nocc")):
            d = df[(df.model == m) & (df.post == post)].set_index("subject_id")["dice"]
            diff = (d - a.loc[d.index, col]).abs().max()
            assert diff < 1e-6, f"{m}/{post}: Dice differs from audit by {diff}"
    print("Self-check: Dice reproduces results/audit_2026-11-15 for all models and post-processing.")

    df["fn_frac_gross"] = df.fn_gross_vox / df.fn_vox.where(df.fn_vox > 0)
    df["fp_frac_gross"] = df.fp_gross_vox / df.fp_vox.where(df.fp_vox > 0)
    for name in LABEL_NAMES.values():
        df[f"miss_rate_{name}"] = df[f"fn_{name}_vox"] / df[f"gt_{name}_vox"].where(df[f"gt_{name}_vox"] > 0)
    df.to_csv(RES / "per_case_metrics.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED)
    srows = []
    for (m, post), g in df.groupby(["model", "post"], sort=False):
        for metric in ("dice", "sdice_1mm", "sdice_2mm", "hd95", "assd"):
            x = g[metric].to_numpy(float)
            lo, hi = boot_ci(x, rng)
            q1, q3 = np.nanpercentile(x, [25, 75])
            srows.append({"model": m, "post": post, "metric": metric, "mean": np.nanmean(x),
                          "ci95_lo": lo, "ci95_hi": hi, "median": np.nanmedian(x), "q1": q1, "q3": q3,
                          "min": np.nanmin(x), "max": np.nanmax(x), "n_nan": int(np.isnan(x).sum())})
    s = pd.DataFrame(srows)
    s.to_csv(RES / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s.round(4).to_string(index=False))


# ── Stage 3: summary plots (no patient images) ──────────────────────────

def stage_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    df = pd.read_csv(RES / "per_case_metrics.csv")
    e = df[(df.model == "ensemble") & (df.post == "cc")].set_index("subject_id")
    en = df[(df.model == "ensemble") & (df.post == "nocc")].set_index("subject_id")
    label_ids = set(e.nsmallest(10, "dice").index) | set((e.dice - en.dice).abs().nlargest(4).index)

    def annotate(ax, x, y):
        for sid in label_ids:
            ax.annotate(sid.replace("BRATS_", ""), (x[sid], y[sid]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.scatter(e.gt_vox / 1000, e.dice, s=18, c="#3b6ea5", label="largest CC (headline)")
    ax.scatter(en.gt_vox / 1000, en.dice, s=10, facecolors="none", edgecolors="#c44e52", label="no CC")
    ax.set_xscale("log"); ax.set_xlabel("GT whole-tumor volume (cm³ = 1000 voxels)"); ax.set_ylabel("Dice")
    rho, p = spearmanr(e.gt_vox, e.dice)
    ax.set_title(f"Dice vs tumor volume (ensemble)\nSpearman ρ = {rho:.2f} (p = {p:.2g}), n = {len(e)}")
    annotate(ax, e.gt_vox / 1000, e.dice); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(e.dice, e.sdice_2mm, s=18, c="#3b6ea5", label="2 mm")
    ax.scatter(e.dice, e.sdice_1mm, s=10, c="#dd8452", alpha=0.7, label="1 mm")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Volumetric Dice"); ax.set_ylabel("Surface Dice")
    ax.set_title("Surface Dice vs volumetric Dice (ensemble, largest CC)")
    annotate(ax, e.dice, e.sdice_2mm); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    delta = e.dice - en.dice
    ax.scatter(e.gt_frac_largest_component, delta, s=18, c="#55a868")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Fraction of GT volume in its largest connected component")
    ax.set_ylabel("Dice(largest CC) − Dice(no CC)")
    ax.set_title("Effect of largest-CC post-processing (ensemble)")
    annotate(ax, e.gt_frac_largest_component, delta); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "summary_plots.png", dpi=150)
    print(f"Saved {RES / 'summary_plots.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    order = e.sort_values("dice").index
    ax.bar(range(len(order)), e.loc[order, "fn_gross_vox"] / 1000, color="#4c72b0", label="FN gross (>2 mm from pred)")
    ax.bar(range(len(order)), e.loc[order, "fn_boundary_vox"] / 1000, bottom=e.loc[order, "fn_gross_vox"] / 1000,
           color="#a1c9f4", label="FN boundary (≤2 mm)")
    ax.bar(range(len(order)), -e.loc[order, "fp_gross_vox"] / 1000, color="#c44e52", label="FP gross (>2 mm from GT)")
    ax.bar(range(len(order)), -e.loc[order, "fp_boundary_vox"] / 1000, bottom=-e.loc[order, "fp_gross_vox"] / 1000,
           color="#ff9f9b", label="FP boundary (≤2 mm)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(order))[:10]); ax.set_xticklabels([s.replace("BRATS_", "") for s in order[:10]],
                                                               rotation=90, fontsize=7)
    ax.set_xlabel("subjects sorted by ensemble Dice (lowest 10 labelled)")
    ax.set_ylabel("error volume (cm³); FN up, FP down")
    ax.set_title("Error volume per case, ensemble + largest CC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RES / "error_volumes.png", dpi=150)
    print(f"Saved {RES / 'error_volumes.png'}")


# ── Stage 4: per-case review panels (contain MRI; kept under outputs/) ───

def review_cases(df):
    e = df[(df.model == "ensemble") & (df.post == "cc")].set_index("subject_id")
    en = df[(df.model == "ensemble") & (df.post == "nocc")].set_index("subject_id")
    low = set(e.index[e.dice < 0.80])
    cc_hit = set(e.index[(e.dice - en.dice).abs() > 0.05])
    return sorted(low | cc_hit, key=lambda s: e.loc[s, "dice"])


def norm(vol):
    fg = vol[vol > 0]
    if fg.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import nibabel as nib

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "per_case_metrics.csv")
    cases = review_cases(df)
    print("Review cases:", ", ".join(cases))
    lab_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.85, 0.2, 0.55), (1.0, 0.85, 0.0, 0.7), (1.0, 0.1, 0.1, 0.8)])
    err_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.85, 0.2, 0.45), (0.15, 0.45, 1.0, 0.85),
                               (1.0, 0.15, 0.15, 0.85), (1.0, 0.55, 0.0, 0.9)])

    for sid in cases:
        img = np.asarray(nib.load(str(RAW / "imagesTr" / f"{sid}.nii.gz")).dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(str(RAW / "labelsTr" / f"{sid}.nii.gz")).dataobj).astype(np.uint8)
        gt = lab > 0
        prob = (np.load(PROB_DIR / "v1" / f"{sid}.npy") + np.load(PROB_DIR / "v2" / f"{sid}.npy")) / 2
        raw = prob >= THR
        cc = largest_cc(raw)
        # error map: 1 TP, 2 FN, 3 FP (kept), 4 TP removed by largest-CC
        err = np.zeros(gt.shape, np.uint8)
        err[cc & gt] = 1
        err[gt & ~raw] = 2
        err[cc & ~gt] = 3
        err[raw & ~cc & gt] = 4
        fp_removed = raw & ~cc & ~gt
        mods = [norm(img[..., c]) for c in range(4)]  # FLAIR, T1, T1gd, T2

        views = []
        z_gt = int(gt.sum((0, 1)).argmax())
        views.append(("axial", z_gt, "max GT"))
        for name, m in (("max FN", err == 2), ("max FP", err == 3), ("max CC-removed TP", err == 4)):
            if m.any():
                z = int(m.sum((0, 1)).argmax())
                if all(abs(z - v[1]) > 3 for v in views if v[0] == "axial"):
                    views.append(("axial", z, name))
        c = np.round(ndi.center_of_mass(gt)).astype(int)
        views += [("coronal", int(c[1]), "GT centroid"), ("sagittal", int(c[0]), "GT centroid")]

        def sl(vol, view, i):
            axis = {"sagittal": 0, "coronal": 1, "axial": 2}[view]
            return np.rot90(np.take(vol, i, axis=axis))

        r = df[(df.subject_id == sid) & (df.model == "ensemble")].set_index("post")
        titles = ["FLAIR", "T1", "T1gd", "T2", "GT labels on FLAIR", "Ensemble errors on FLAIR"]
        fig, axes = plt.subplots(len(views), 6, figsize=(18, 3.1 * len(views)))
        fig.patch.set_facecolor("#111111")
        for row, (view, i, why) in enumerate(views):
            for col in range(6):
                ax = axes[row, col]
                base = mods[col] if col < 4 else mods[0]
                ax.imshow(sl(base, view, i), cmap="gray", vmin=0, vmax=1, interpolation="none")
                if col == 4:
                    ax.imshow(sl(lab, view, i), cmap=lab_cmap, vmin=0, vmax=3, interpolation="none")
                if col == 5:
                    ax.imshow(sl(err, view, i), cmap=err_cmap, vmin=0, vmax=4, interpolation="none")
                    if fp_removed.any():
                        ax.contour(sl(fp_removed, view, i), levels=[0.5], colors=["#ff66ff"], linewidths=0.6)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if row == 0:
                    ax.set_title(titles[col], color="w", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{view} {i}\n({why})", color="w", fontsize=9)
        fig.suptitle(
            f"{sid}  |  GT {int(gt.sum()):,} vox (edema {int((lab == 1).sum()):,}, non-enh {int((lab == 2).sum()):,}, "
            f"enh {int((lab == 3).sum()):,})  |  ensemble Dice CC {r.loc['cc', 'dice']:.3f} / no-CC {r.loc['nocc', 'dice']:.3f}"
            f"  |  sDice@2mm {r.loc['cc', 'sdice_2mm']:.3f}  HD95 {r.loc['cc', 'hd95']:.1f} mm\n"
            "labels: green edema, yellow non-enhancing, red enhancing   |   errors: green TP, blue FN, red FP, "
            "orange TP removed by largest-CC, magenta outline FP removed by largest-CC",
            color="w", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(FIG_DIR / f"{sid}.png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  saved {sid}.png ({len(views)} views)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["infer", "metrics", "plots", "figures"])
    stage = ap.parse_args().stage
    {"infer": stage_infer, "metrics": stage_metrics, "plots": stage_plots, "figures": stage_figures}[stage]()
