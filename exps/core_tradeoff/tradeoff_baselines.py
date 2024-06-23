from typing import List
import os

from exps.plot_baselines import create_tradeoff_plots
from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    UNIFORM_PAPER,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,
    COR_NORMAL_CUSTOM,
    CLEAR_LEADER_HARD,
    NO_CLEAR_LEADER,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    HNSW_MIPS,
    NAPG_MIPS,
    NEQ_MIPS,
    BUCKET_ACTION_ELIMINATION,
    H2ALSH,
    ASYMMETRIC,

    # tradeoff specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
    CRYPTO_PAIRS_SIGNAL_LENGTH,
)


def get_tradeoff_params(data_type):
    """
    Function gives the constants for [num_atoms, len_signals] for the specified datatype.
    """
    if data_type in [NETFLIX, MOVIE_LENS, CRYPTO_PAIRS]:
        num_atoms = NUM_ATOMS_REAL
    else:
        num_atoms = TRADEOFF_NUM_ATOMS

    if data_type is NETFLIX:
        len_signals = LEN_SIGNAL_NETFLIX
    elif data_type is MOVIE_LENS:
        len_signals = LEN_SIGNAL_MOVIE
    elif data_type is CRYPTO_PAIRS:
        len_signals = CRYPTO_PAIRS_SIGNAL_LENGTH
    else:
        len_signals = TRADEOFF_DIMENSION

    return num_atoms, len_signals


def large_dim_tradeoff_baselines(is_plot: bool = False):
    # Large dims tradeoff plot for NEQ, BANDIT_MIPS algorithms
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        NEQ_MIPS,
    ]
    data_types = [NETFLIX, NORMAL_CUSTOM, COR_NORMAL_CUSTOM]
    len_signals = 10 ** 5
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "large_dim_logs", f"topk_{top_k}")
        for data_type in data_types:
            for alg in algorithms:
                speedup_precision_exps(
                    num_atoms=TRADEOFF_NUM_ATOMS,
                    num_experiments=TRADEOFF_NUM_EXPERIMENTS,
                    len_signals=len_signals,
                    num_signals=30,
                    data_type=data_type,
                    mips_alg=alg,
                    with_replacement=False,
                    dir_name=log_dir,
                    num_best_atoms=top_k,
                    num_seeds=1,
                )
        if is_plot:
            create_tradeoff_plots(
                alg_names=algorithms,
                data_types=data_types,
                top_k=top_k,
                log_dir=log_dir,
                max_speedup=None,
                is_logspace=True,
            )


def tradeoff_baselines(is_plot: bool = False):
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        GREEDY_MIPS,
        LSH_MIPS,
        H2ALSH,
        NEQ_MIPS,
        PCA_MIPS,
        NAPG_MIPS,
        HNSW_MIPS,
    ]
    data_types = [MOVIE_LENS, NETFLIX, NORMAL_CUSTOM, COR_NORMAL_CUSTOM]

    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "normalized_logs", f"topk_{top_k}")
        for data_type in data_types:  # NORMAL_CUSTOM,
            if data_type is NETFLIX:
                len_signals = LEN_SIGNAL_NETFLIX
            elif data_type is MOVIE_LENS:
                len_signals = LEN_SIGNAL_MOVIE
            else:
                len_signals = TRADEOFF_DIMENSION
            for alg in algorithms:
                speedup_precision_exps(
                    num_atoms=TRADEOFF_NUM_ATOMS,
                    num_experiments=TRADEOFF_NUM_EXPERIMENTS,
                    len_signals=len_signals,
                    num_signals=30,
                    data_type=data_type,
                    mips_alg=alg,
                    with_replacement=False,
                    dir_name=log_dir,
                    num_best_atoms=top_k,
                    num_seeds=1,
                )
        if is_plot:
            create_tradeoff_plots(
                alg_names=algorithms,
                data_types=data_types,
                top_k=top_k,
                log_dir=log_dir,
                max_speedup=None,
                is_logspace=True,
            )


def tradeoff_baselines_plot():
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        GREEDY_MIPS,
        LSH_MIPS,
        H2ALSH,
        NEQ_MIPS,
        PCA_MIPS,
        NAPG_MIPS,
        HNSW_MIPS,
    ]
    data_types = [MOVIE_LENS, NETFLIX, NORMAL_CUSTOM, COR_NORMAL_CUSTOM]
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "normalized_logs", f"topk_{top_k}")
        create_tradeoff_plots(
            alg_names=algorithms,
            data_types=data_types,
            top_k=top_k,
            log_dir=log_dir,
            max_speedup=None,
            is_logspace=True,
        )


if __name__ == "__main__":
    # tradeoff_baselines()
    tradeoff_baselines_plot()