import numpy as np
import os
from typing import Tuple
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import generate_scaling_plots, create_scaling_plots
from utils.constants import (
    BATCH_SIZE,
    SCALING_BASELINES_DATATYPES,
    SCALING_BASELINES_ALGORITHMS,

    # datasets with specific params
    UNIFORM_PAPER,
    CRYPTO_PAIRS,
    MOVIE_LENS,
    NORMAL_CUSTOM,
    COR_NORMAL_CUSTOM,
    NETFLIX,
    NORMAL_PAPER,
    HIGHLY_SYMMETRIC,

    # scaling specific constants
    ACTION_ELIMINATION,
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_ATOMS_CRYPTO_PAIRS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
    DIMENSION_OF_ATOMS,
    LARGE_DELTA,
    DEFAULT_MAXMIN,
)


def get_scaling_baseline_params(data_type):
    """
    Function gives the constants for [epsilon, delta, size_minmax] for the specified datatype.
    This is for bandit-MIPS ONLY!!!
    """
    if data_type == NORMAL_CUSTOM:
        # epsilon, delta, maxmin, size_minmax
        return 0.001, 0.001, (5, 0), (10 ** 4, 10 ** 6)
    elif data_type == COR_NORMAL_CUSTOM:
        # epsilon, delta, maxmin, size_minmax
        return 0.001, 0.001, (5, 0), (10 ** 4, 10 ** 6)
    elif data_type == NETFLIX:
        return 0.01, 0.01, (2, 0), (10 ** 4, 143458)
    elif data_type == MOVIE_LENS:
        # epsilon, delta, maxmin, size_minmax
        return 0.001, 0.01, (1, 0), (1000, 5000)
    elif data_type == NORMAL_PAPER:
        # epsilon, delta, maxmin, size_minmax
        return 0.001, 0.01, (10, 0), (10 ** 3, 10 ** 5)
    else:
        print(f"{data_type} shouldn't be generated here")

    return epsilon, delta, maxmin, size_minmax


def scaling_baselines(
        algorithms, 
        data_types,
        dir_name,
    ):
    """
    Run scaling experiments for the 5 datasets (2 synthetic, 3 real) on 9 baseline MIPS algorithms.
    This function is called by repro_script_python.py.
    """
    if os.path.exists(dir_name):
        print(f"=> retrieving logs...")
    else:
        # Get data for the scaling experiments for the datasets (not the high-dimensional datasets)
        for data_type in data_types:
            for algorithm in algorithms:
                print(f"=> Creating scaling log files for {algorithm} on {data_type}")
                epsilon, delta, maxmin, size_minmax = get_scaling_baseline_params(data_type)
                scaling_exp(
                    epsilon=epsilon,
                    delta=delta,
                    maxmin=maxmin,
                    num_atoms=SCALING_NUM_ATOMS,
                    size_minmax=size_minmax,
                    num_experiments=SCALING_NUM_EXPERIMENTS,
                    num_signals=SCALING_NUM_SIGNALS,
                    data_type=data_type,
                    mips_alg=algorithm,
                    with_replacement=False,
                    is_normalize=False,
                    is_logspace=False,
                    num_best_atoms=1,
                    add_noise=False,
                    dir_name=dir_name,
                )


if __name__ == "__main__":
    scaling_baselines([ACTION_ELIMINATION])