import os

from utils.constants import (
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
    NETFLIX,
    DIMENSION_OF_ATOMS,
)
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots


def run_time_comparison():
    # Hard-coded
    epsilon = 0.01
    delta = 0.01
    maxmin = (0, 5)
    num_experiments = 15
    num_seeds = 3
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
    ]
    number_of_atoms = 100
    top_k = 1

    exp_dir = os.path.dirname(__file__)
    log_dir = os.path.join(exp_dir, "logs")
    for algorithm in algorithms:

        # Compile numba decorated function
        scaling_exp(
            mips_alg=algorithm,
            size_minmax=(100, 100),
            num_experiments=1,
            num_seeds=1,
            is_log=False,
        )

        # Real Experiment
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            maxmin=maxmin,
            num_experiments=num_experiments,
            num_seeds=num_seeds,
            mips_alg=algorithm,
            size_minmax=(100, 100000),
            independent_var=DIMENSION_OF_ATOMS,
            num_atoms=number_of_atoms,
            num_best_atoms=top_k,
            is_logspace=True,
            data_type=NETFLIX,
            is_log=True,
            dir_name=log_dir,
        )


def plot_time_comparison():
    exp_dir = os.path.dirname(__file__)
    log_dir = os.path.join(exp_dir, "logs")

    create_scaling_plots(
        is_fit=False,
        include_error_bar=False,
        data_types=[NETFLIX],
        alg_names=[ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION, MEDIAN_ELIMINATION],
        dir_name=log_dir,
        is_logspace_y=False,
        is_logspace_x=True,
        is_plot_runtime=True,
        is_plot_accuracy=True,
    )


if __name__ == "__main__":
    run_time_comparison()
    plot_time_comparison()
