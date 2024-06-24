import numpy as np
import math
import os
from typing import Tuple

from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    BATCH_SIZE,
    # datasets
    SIFT_1M,
    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_EPSILON,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_EPSILON_REAL,
)


def sift_scaling(
    run: bool = True, 
    plot: bool = True,
    save_to: str = "",
):
    """
    Run scaling experiments for the toy datasets
    """
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(parent_dir, "sift_scaling_logs")
    if run:
        if os.path.exists(os.path.join(parent_dir, dirname)):
            print("=> sift 1m logs exists!")
        else:
            for data_type in [SIFT_1M]:
                mean_acc = scaling_exp(
                    epsilon=0.00,
                    delta=0.000001,
                    num_atoms=10,
                    size_minmax=(1e5, 1e6),
                    maxmin=(5000, 0),
                    num_experiments=10,
                    num_signals=1,
                    data_type=SIFT_1M,
                    mips_alg=ACTION_ELIMINATION,
                    with_replacement=True,
                    is_normalize=False,
                    is_logspace=True,
                    dir_name=dirname,
                    num_seeds=5,
                )
                # TODO(@motiwari): Figure out why this is not 100%
                assert (
                    mean_acc >= 0.95
                ), f"Mean accuracy on dataset {data_type} should be above 95\%"

    if plot:
        create_scaling_plots(
            data_types=[SIFT_1M],
            alg_names=[ACTION_ELIMINATION],
            include_error_bar=True,
            is_fit=True,
            is_logspace_x=False,
            is_logspace_y=False,
            dir_name=dirname,
            save_to=save_to,
        )


if __name__ == "__main__":
    sift_scaling(run=True, plot=True)
