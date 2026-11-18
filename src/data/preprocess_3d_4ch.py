"""Preprocess MSD Task01 into 4-channel volumes for the E1 CV run (results/RUNS.md, E1).

Channel order (C, H, W, D): 0 FLAIR, 1 T1gd, 2 T1w, 3 T2w. Channels 0 and 1 are the
same inputs as data/processed_3d (E0); 2 and 3 are added. Each channel uses the same
per-channel normalization as preprocess_3d.py (foreground 1st/99th percentile clip,
min-max to [0, 1]).

Stored as float16: the network's first layer is a Conv3d run under autocast, which
casts its float32 input to float16 with the same round-to-nearest-even, so the
network sees bit-identical inputs (checked against processed_3d by --check).
Masks are not duplicated; use data/processed_3d/masks.

    python src/data/preprocess_3d_4ch.py            # writes data/processed_3d_4ch/images
    python src/data/preprocess_3d_4ch.py --check    # verify channels 0-1 vs processed_3d
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocess_3d import normalize_channel  # identical normalization to E0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour" / "imagesTr"
E0_IMG = PROJECT_ROOT / "data" / "processed_3d" / "images"
OUT = PROJECT_ROOT / "data" / "processed_3d_4ch" / "images"
ORDER = (0, 2, 1, 3)  # raw MSD channels: 0 FLAIR, 1 T1w, 2 t1gd, 3 T2w


def build(sid):
    out = OUT / f"{sid}.npy"
    if out.exists():
        return sid, "exists"
    img4d = nib.load(str(RAW / f"{sid}.nii.gz")).get_fdata()
    vol = np.stack([normalize_channel(img4d[..., c]) for c in ORDER], axis=0).astype(np.float16)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, vol)
    tmp.replace(out)
    return sid, "written"


def check(sid):
    new = np.load(OUT / f"{sid}.npy")
    old = np.load(E0_IMG / f"{sid}.npy")
    assert new.shape[1:] == old.shape[1:] and new.shape[0] == 4, sid
    return sid, bool(np.array_equal(new[:2], old.astype(np.float16)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    sids = sorted(f.stem for f in E0_IMG.glob("*.npy"))
    assert len(sids) == 484
    OUT.mkdir(parents=True, exist_ok=True)
    fn = check if args.check else build
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(fn, sids, chunksize=4))
    if args.check:
        bad = [s for s, ok in res if not ok]
        print(f"channels 0-1 identical to fp16(processed_3d): {len(sids) - len(bad)}/{len(sids)}", bad[:10])
        sys.exit(1 if bad else 0)
    print({k: sum(1 for _, s in res if s == k) for k in ("written", "exists")})


if __name__ == "__main__":
    main()
