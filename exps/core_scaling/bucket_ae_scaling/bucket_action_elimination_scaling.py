import os
from typing import Tuple
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import generate_scaling_plots, create_scaling_plots
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
    POSITIVE_COR_NORMAL_CUSTOM,
    CLEAR_LEADER_HARD,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    BUCKET_ACTION_ELIMINATION,

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
    BUCKET_AE_NUM_SEEDS,
)


def scaling_bucket_ae():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(this_dir, "logs")
    for dataset in [NETFLIX_TRANSPOSE, NORMAL_CUSTOM, POSITIVE_COR_NORMAL_CUSTOM]:
        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=NUMBER_OF_ATOMS,
            signal_dim=1000,
            size_minmax=(100, 10 ** 5),
            num_best_atoms=1,
            num_experiments=20,
            num_signals=2,
            data_type=dataset,
            mips_alg=BUCKET_ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            dir_name=log_dir,
            num_seeds=BUCKET_AE_NUM_SEEDS,
        )

        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=NUMBER_OF_ATOMS,
            signal_dim=1000,
            size_minmax=(100, 10 ** 5),
            num_best_atoms=1,
            num_experiments=20,
            num_signals=2,
            data_type=dataset,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            dir_name=log_dir,
            num_seeds=BUCKET_AE_NUM_SEEDS
        )
    for dataset in [NETFLIX, NORMAL_CUSTOM, POSITIVE_COR_NORMAL_CUSTOM]:
        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=DIMENSION_OF_ATOMS,
            num_atoms=1000,
            signal_dim=3000,
            size_minmax=(1000, 5 * 10 ** 4),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=1,
            data_type=dataset,
            mips_alg=BUCKET_ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            dir_name=log_dir,
            num_seeds=BUCKET_AE_NUM_SEEDS,
        )

        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=DIMENSION_OF_ATOMS,
            num_atoms=1000,
            signal_dim=3000,
            size_minmax=(1000, 5 * 10 ** 4),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=1,
            data_type=dataset,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            dir_name=log_dir,
            num_seeds=BUCKET_AE_NUM_SEEDS,
        )


def scaling_bucket_ae_plot():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(this_dir, "logs")
    datasets = [NETFLIX, NETFLIX_TRANSPOSE, NORMAL_CUSTOM, POSITIVE_COR_NORMAL_CUSTOM]
    create_scaling_plots(
        data_types=datasets,
        alg_names=[ACTION_ELIMINATION],
        ind_variables=[NUMBER_OF_ATOMS, DIMENSION_OF_ATOMS],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=True,
        dir_name=log_dir,
    )
    create_scaling_plots(
        data_types=datasets,
        alg_names=[BUCKET_ACTION_ELIMINATION],
        ind_variables=[NUMBER_OF_ATOMS, DIMENSION_OF_ATOMS],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=True,
        dir_name=log_dir,
    )
    create_scaling_plots(
        data_types=datasets,
        alg_names=[ACTION_ELIMINATION, BUCKET_ACTION_ELIMINATION],
        ind_variables=[NUMBER_OF_ATOMS, DIMENSION_OF_ATOMS],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=False,
        dir_name=log_dir
    )


if __name__ == "__main__":
    # scaling_bucket_ae()
    scaling_bucket_ae_plot()