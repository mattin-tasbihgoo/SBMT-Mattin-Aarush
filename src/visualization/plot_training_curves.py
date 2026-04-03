from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "flair_t1gd_25d5_binary"
LOG_CSV = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME / "logs" / "training_log.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not LOG_CSV.exists():
        raise FileNotFoundError(f"Missing training log: {LOG_CSV}")

    df = pd.read_csv(LOG_CSV).sort_values("epoch").reset_index(drop=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(df["epoch"], df["train_loss"], marker="o")
    ax[0].set_title("Train Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(df["epoch"], df["val_loss"], marker="o")
    ax[1].set_title("Validation Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")

    ax[2].plot(df["epoch"], df["val_dice"], marker="o", label="All slices")
    if "val_pos_dice" in df.columns:
        ax[2].plot(df["epoch"], df["val_pos_dice"], marker="s", label="Positive only")
        ax[2].legend()
    ax[2].set_title("Validation Dice")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Dice")

    plt.tight_layout()
    out_path = FIG_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()