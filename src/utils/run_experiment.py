import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import main as train_main
from visualization.plot_training_curves import main as plot_main
from evaluation.evaluate import main as eval_main

RUN_PLOTS = True
RUN_EVAL = True

if __name__ == "__main__":
    train_main()

    if RUN_PLOTS:
        plot_main()

    if RUN_EVAL:
        eval_main()