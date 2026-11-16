"""Run the whole E0 CV baseline: train + score folds 0..4 in turn, then aggregate.

Resumable: finished folds are skipped, an interrupted fold resumes from its
last epoch. Keeps Windows awake while running (process-scoped; no system
setting is changed).

    python src/training/run_cv.py --run-dir outputs/cv_e0_v1recipe_2026-11-15 --tag cv_e0_2026-11-15
"""

import argparse
import ctypes
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"


def stamp(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(cmd, fatal=True):
    stamp("run: " + " ".join(map(str, cmd)))
    rc = subprocess.run([sys.executable, *map(str, cmd)], cwd=PROJECT_ROOT).returncode
    if rc != 0:
        stamp(f"FAILED (exit {rc})")
        if fatal:
            sys.exit(rc)
    return rc


def train_fold(k, run_dir, workers):
    # A crash (e.g. Windows commit limit hit by data workers) resumes from the last
    # finished epoch with fewer workers; the recipe is unchanged.
    for attempt in range(4):
        w = max(4, workers - 4 * attempt)
        rc = run([SRC / "training" / "train_cv.py", "--fold", k, "--run-dir", run_dir,
                  "--workers", w], fatal=False)
        if rc == 0:
            return
        stamp(f"fold {k}: retry {attempt + 1} with fewer workers")
    stamp(f"fold {k}: giving up")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process lives
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    for k in range(5):
        train_fold(k, args.run_dir, args.workers)
        run([SRC / "evaluation" / "evaluate_cv.py", "--fold", k, "--run-dir", args.run_dir])
    run([SRC / "evaluation" / "cv_report.py", "--run-dir", args.run_dir, "--tag", args.tag])
    stamp("ALL DONE")


if __name__ == "__main__":
    main()
