"""Create the frozen 5-fold cross-validation split (results/PROTOCOL.md).

All 484 labeled MSD Task01 subjects are assigned to exactly one of 5 folds,
stratified by whole-tumor volume: subjects are sorted by tumor voxel count,
cut into consecutive blocks of 5, and each block is spread over the 5 folds
in a seeded random order. Run once; the output file is committed and frozen.
Refuses to overwrite an existing folds file.

    python src/training/make_cv_folds.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed_3d"
OUT = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"
APRIL_VAL = (PROJECT_ROOT / "outputs" / "runtime_flair_t1gd_25d5_smalllesion_baseline_v1"
             / "logs" / "val_subjects.txt")
N_FOLDS = 5
SEED = 20261115


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists; the split is frozen. Delete it by hand only if you mean to.")
    meta = pd.read_csv(DATA / "metadata.csv")[["subject_id", "tumor_voxels"]]
    assert len(meta) == 484 and meta.subject_id.is_unique

    # Re-count tumor voxels from the masks instead of trusting metadata.csv.
    counts = []
    for sid in meta.subject_id:
        counts.append(int(np.load(DATA / "masks" / f"{sid}.npy").astype(bool).sum()))
    meta["mask_voxels"] = counts
    bad = meta[meta.mask_voxels != meta.tumor_voxels]
    assert bad.empty, f"metadata tumor_voxels disagree with masks:\n{bad}"
    assert (meta.mask_voxels > 0).all()

    rng = np.random.default_rng(SEED)
    meta = meta.sort_values(["mask_voxels", "subject_id"]).reset_index(drop=True)
    folds = np.empty(len(meta), dtype=int)
    for start in range(0, len(meta), N_FOLDS):
        block = np.arange(start, min(start + N_FOLDS, len(meta)))
        folds[block] = rng.permutation(N_FOLDS)[: len(block)]
    meta["fold"] = folds

    april_val = {l.strip() for l in APRIL_VAL.read_text().splitlines() if l.strip()}
    meta["april_val96"] = meta.subject_id.isin(april_val).astype(int)
    assert meta.april_val96.sum() == 96

    out = meta.sort_values("subject_id")[["subject_id", "fold", "mask_voxels", "april_val96"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, lineterminator="\n")

    print(out.groupby("fold").agg(n=("subject_id", "size"),
                                  median_vox=("mask_voxels", "median"),
                                  mean_vox=("mask_voxels", "mean"),
                                  april_val96=("april_val96", "sum")))
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
