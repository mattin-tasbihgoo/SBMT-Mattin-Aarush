"""Train one fold of the 5-fold CV protocol (results/PROTOCOL.md).

Recipe E0 = the April V1 recipe (src/training/train_3d.py) with these
protocol changes only:
  * subjects come from the frozen folds file: train on the 4 other folds,
    never touch the test fold here;
  * no per-epoch validation and no checkpoint selection: the checkpoint is
    the final epoch of the fixed 50-epoch cosine schedule;
  * full training state is saved every epoch so an interrupted run resumes.

Same model (UNet3D [32,64,128,256], 2 channels), patches (128^3, 4/subject,
50% foreground-centred, random flips), batch 4, Adam 1e-4, cosine T_max=50
eta_min 1e-6, BCE+Dice 0.5/0.5, grad clip 1.0, AMP, seed 42.

    python src/training/train_cv.py --fold 0 --run-dir outputs/cv_e0_v1recipe_2026-11-15
    python src/training/train_cv.py --fold 0 --run-dir ... --bench 40   # timing only, writes nothing
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset_3d import BrainTumorDataset3D
from models.model_3d import UNet3D
from training.train_3d import combined_loss, dice_scores, set_seed  # identical loss to April V1
from evaluation.evaluate_3d import git_commit, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d"
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
FOLDS_CSV = PROJECT_ROOT / "results" / "protocol" / "folds_5fold_seed20261115.csv"

RECIPE = {
    "name": "E0_v1_recipe",
    "model": "UNet3D features [32,64,128,256]",
    "in_channels": 2,
    "channels": "FLAIR, T1gd (processed_3d channels 0,1)",
    "patch_size": [128, 128, 128],
    "patches_per_subject": 4,
    "foreground_bias": 0.5,
    "augmentation": "random flips on 3 axes",
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-4,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR T_max=50 eta_min=1e-6",
    "loss": "0.5 BCE + 0.5 soft Dice",
    "grad_clip": 1.0,
    "amp": True,
    "seed": 42,
    "checkpoint_rule": "final epoch (no selection)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bench", type=int, default=0, help="time N iterations and exit; writes nothing")
    return p.parse_args()


def fold_subjects(fold: int):
    df = pd.read_csv(FOLDS_CSV)
    assert len(df) == 484 and df.subject_id.is_unique
    train = sorted(df.loc[df.fold != fold, "subject_id"])
    test = sorted(df.loc[df.fold == fold, "subject_id"])
    assert not set(train) & set(test)
    return train, test


def main():
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    fold_dir = run_dir / f"fold{args.fold}"
    final_ckpt = fold_dir / "final_model.pt"
    state_path = fold_dir / "state_latest.pt"
    if final_ckpt.exists() and not args.bench:
        print(f"{final_ckpt} exists; fold {args.fold} already trained. Nothing to do.")
        return

    set_seed(RECIPE["seed"])
    train_subjects, test_subjects = fold_subjects(args.fold)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_ds = BrainTumorDataset3D(
        train_subjects, str(IMG_DIR), str(MSK_DIR),
        patch_size=tuple(RECIPE["patch_size"]),
        patches_per_subject=RECIPE["patches_per_subject"],
        mode="train", augment=True,
    )
    loader = DataLoader(train_ds, batch_size=RECIPE["batch_size"], shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    model = UNet3D(in_channels=2, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=RECIPE["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=RECIPE["epochs"], eta_min=1e-6)
    scaler = GradScaler(device=device)

    if args.bench:
        model.train()
        t0 = None
        for i, (x, y, _) in enumerate(loader):
            if i == 5:
                torch.cuda.synchronize(); t0 = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device):
                loss = combined_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            if i == 5 + args.bench:
                break
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.bench
        print(f"fold {args.fold}: {dt:.3f} s/iter, {len(loader)} iters/epoch -> "
              f"{dt * len(loader) / 60:.1f} min/epoch, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        return

    fold_dir.mkdir(parents=True, exist_ok=True)
    history, start_epoch = [], 0
    if state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        history, start_epoch = st["history"], st["epoch"]
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["np_rng"])
        print(f"Resuming fold {args.fold} after epoch {start_epoch}")
        with open(fold_dir / "resumed.txt", "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} resumed after epoch {start_epoch}\n")
    else:
        (fold_dir / "train_subjects.txt").write_text("\n".join(train_subjects) + "\n")
        (fold_dir / "test_subjects.txt").write_text("\n".join(test_subjects) + "\n")
        cfg = {"fold": args.fold, "recipe": RECIPE, "n_train": len(train_subjects),
               "n_test": len(test_subjects), "folds_csv": str(FOLDS_CSV.relative_to(PROJECT_ROOT)),
               "folds_sha256": sha256(FOLDS_CSV), "git_commit": git_commit(),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "workers": args.workers,
               "started": datetime.now().isoformat(timespec="seconds")}
        (fold_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))

    print(f"Fold {args.fold}: {len(train_subjects)} train subjects, {len(loader)} iters/epoch, "
          f"epochs {start_epoch + 1}..{RECIPE['epochs']}", flush=True)

    for epoch in range(start_epoch, RECIPE["epochs"]):
        t0 = time.time()
        model.train()
        loss_sum = dice_sum = seen = 0.0
        for i, (x, y, _) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device):
                logits = model(x)
                loss = combined_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            loss_sum += loss.item() * bs
            dice_sum += dice_scores(logits, y).sum().item()
            seen += bs
            if i % 50 == 0:
                print(f"fold {args.fold} epoch {epoch + 1}/{RECIPE['epochs']} iter {i}/{len(loader)} "
                      f"loss {loss_sum / seen:.4f}", flush=True)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "train_dice": dice_sum / seen,
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_sec": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(fold_dir / "training_log.csv", index=False)
        tmp = fold_dir / "state_latest.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "history": history,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}, tmp)
        tmp.replace(state_path)
        print(f"fold {args.fold} epoch {epoch + 1}/{RECIPE['epochs']} done | loss {row['train_loss']:.4f} "
              f"dice {row['train_dice']:.4f} | lr {row['lr']:.2e} | {row['epoch_time_sec']:.0f}s", flush=True)

    torch.save(model.state_dict(), final_ckpt)
    print(f"fold {args.fold} finished: {final_ckpt}", flush=True)


if __name__ == "__main__":
    main()
