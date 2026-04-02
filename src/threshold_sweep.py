from pathlib import Path
import pandas as pd

from evaluate import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "flair_t1gd_binary"

OUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
SWEEP_DIR = OUT_ROOT / "eval" / "threshold_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sweep configuration
# -----------------------------
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Try both raw and cleaned predictions
POSTPROCESS_SETTINGS = [
    {"use_postprocess": False, "min_component_size": 0, "keep_largest_n": None},
    {"use_postprocess": True, "min_component_size": 20, "keep_largest_n": None},
    {"use_postprocess": True, "min_component_size": 30, "keep_largest_n": None},
    {"use_postprocess": True, "min_component_size": 50, "keep_largest_n": None},
]

# Set to True only if you want lots of overlay folders written out
SAVE_OVERLAYS_DURING_SWEEP = False


def main():
    rows = []

    for pp_cfg in POSTPROCESS_SETTINGS:
        for thr in THRESHOLDS:
            print("=" * 80)
            print(
                f"Running threshold={thr:.2f}, "
                f"use_postprocess={pp_cfg['use_postprocess']}, "
                f"min_component_size={pp_cfg['min_component_size']}, "
                f"keep_largest_n={pp_cfg['keep_largest_n']}"
            )

            summary = run_evaluation(
                threshold=thr,
                use_postprocess=pp_cfg["use_postprocess"],
                min_component_size=pp_cfg["min_component_size"],
                keep_largest_n=pp_cfg["keep_largest_n"],
                save_overlays=SAVE_OVERLAYS_DURING_SWEEP,
            )
            rows.append(summary)

    df = pd.DataFrame(rows)

    summary_csv = SWEEP_DIR / "threshold_sweep_results.csv"
    df.to_csv(summary_csv, index=False)

    # Sort views for quick inspection
    by_pos_slice = df.sort_values(
        ["positive_slice_dice", "positive_subject_dice", "all_slice_dice"],
        ascending=False
    ).reset_index(drop=True)

    by_pos_subject = df.sort_values(
        ["positive_subject_dice", "positive_slice_dice", "all_subject_dice"],
        ascending=False
    ).reset_index(drop=True)

    top_pos_slice_csv = SWEEP_DIR / "threshold_sweep_sorted_by_positive_slice_dice.csv"
    top_pos_subject_csv = SWEEP_DIR / "threshold_sweep_sorted_by_positive_subject_dice.csv"

    by_pos_slice.to_csv(top_pos_slice_csv, index=False)
    by_pos_subject.to_csv(top_pos_subject_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Saved full sweep results to: {summary_csv}")
    print(f"Saved sorted (positive slice Dice) to: {top_pos_slice_csv}")
    print(f"Saved sorted (positive subject Dice) to: {top_pos_subject_csv}")

    print("\nTop 10 by positive slice Dice:")
    print(
        by_pos_slice[
            [
                "setting_tag",
                "threshold",
                "use_postprocess",
                "min_component_size",
                "keep_largest_n",
                "positive_slice_dice",
                "positive_subject_dice",
                "all_slice_dice",
                "all_subject_dice",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nBest setting by positive slice Dice:")
    print(by_pos_slice.iloc[0].to_dict())

    print("\nBest setting by positive subject Dice:")
    print(by_pos_subject.iloc[0].to_dict())


if __name__ == "__main__":
    main()