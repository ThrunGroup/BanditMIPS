import math
import os.path
from typing import Tuple
from exps.speedup_precision_exps import speedup_precision_exps
from exps.plot_baselines import create_tradeoff_plots
from utils.constants import (
    BATCH_SIZE,
    NUMBER_OF_ATOMS,
    DIMENSION_OF_ATOMS,
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    NETFLIX_TRANSPOSE,
    MOVIE_LENS,
    COR_NORMAL_CUSTOM,
    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    BUCKET_ACTION_ELIMINATION,
    H2ALSH,
    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
    TRADEOFF_NUM_EXPERIMENTS,
)


def h2alsh_tradeoff_experiment():
    """
    Run speedup_precision tradeoff experiment for H2ALSH algorithm and a few other MIPS algorithms with Normal custom
    dataset
    """
    num_atoms = 100000
    num_experiments = TRADEOFF_NUM_EXPERIMENTS
    len_signals = 100
    num_signals = 3
    top_k = 3
    num_seeds = 2
    parent_dir = os.path.dirname(__file__)
    log_dir = os.path.join(parent_dir, "logs", "h2alsh_precision_speed")
    datasets = [NORMAL_CUSTOM] #, COR_NORMAL_CUSTOM, NETFLIX]

    for data_type in datasets:
        for mips_alg in [
            # ADAPTIVE_ACTION_ELIMINATION,
            # ACTION_ELIMINATION,
            BUCKET_ACTION_ELIMINATION,
            H2ALSH,
            # GREEDY_MIPS,
        ]:
            if data_type is NETFLIX:
                num_atoms = 1300
            speedup_precision_exps(
                num_atoms=num_atoms,
                num_experiments=num_experiments,
                len_signals=len_signals,
                num_signals=num_signals,
                data_type=data_type,
                mips_alg=mips_alg,
                with_replacement=False,
                num_best_atoms=top_k,
                num_seeds=num_seeds,
                dirname=log_dir,
            )

    create_tradeoff_plots(
        data_types=datasets,
        alg_names=[
            # ADAPTIVE_ACTION_ELIMINATION,
            # ACTION_ELIMINATION,
            BUCKET_ACTION_ELIMINATION,
            H2ALSH,
            # GREEDY_MIPS,
        ],
        top_k=top_k,
        # max_speedup=10,
        dir=log_dir,
    )


if __name__ == "__main__":
    h2alsh_tradeoff_experiment()
