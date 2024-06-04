from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
import os

from utils.constants import (

    # datasets
    NORMAL_CUSTOM,

    # algorithms
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,

    # scaling specific constants
    LARGE_SIZE_MINMAX,
    LARGE_NUM_ATOMS,
    LARGE_NUM_EXPERIMENTS,
    LARGE_DELTA,
    LARGE_EPSILON,
    LARGE_MAXMIN
)


def large_scaling_exps():
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
    log_dir = os.path.join(os.path.join(root_dir, "logs"), "large_scaling")
    if not os.path.exists(log_dir):
        for model in [ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION]:
            scaling_exp(
                mips_alg=model,
                size_minmax=LARGE_SIZE_MINMAX,
                num_atoms=LARGE_NUM_ATOMS,
                is_logspace=True,
                num_experiments=LARGE_NUM_EXPERIMENTS,
                num_seeds=10,
                data_type=NORMAL_CUSTOM,
                maxmin=LARGE_MAXMIN,
                delta=LARGE_DELTA,
                epsilon=LARGE_EPSILON,
                dirname=log_dir,
            )
    else:
        print("=> Large Scaling experiments log files already exist.")
    print("=> Generating large scale experiments plots")
    create_scaling_plots(
        alg_names=[ACTION_ELIMINATION],
        data_types=[NORMAL_CUSTOM],
        is_logspace=True,
        is_fit=True,
        is_plot_accuracy=True,
        include_error_bar=True,
        dirname=log_dir,
    )
    create_scaling_plots(
        alg_names=[ADAPTIVE_ACTION_ELIMINATION],
        data_types=[NORMAL_CUSTOM],
        is_logspace=True,
        is_fit=True,
        is_plot_accuracy=True,
        include_error_bar=True,
        dirname=log_dir,
    )


if __name__ == "__main__":
    large_scaling_exps()

