from train import main as train_main
from plot_training_curves import main as plot_main
from evaluate import main as eval_main

RUN_PLOTS = True
RUN_EVAL = True

if __name__ == "__main__":
    train_main()

    if RUN_PLOTS:
        plot_main()

    if RUN_EVAL:
        eval_main()