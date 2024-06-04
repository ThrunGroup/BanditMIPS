"""
The main objective of this separate neq_plot function is that it is used to compare with our bandit-MIPS algorithm.
If you want to plot all the baselines at once, run plot_baselines.py.
"""

from exps.plot_baselines import create_scaling_plots, create_tradeoff_plots
from utils.constants import (
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,

    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    NEQ_MIPS,

    MAX_SPEEDUP,
    TRADEOFF_TOPK,
)


if __name__ == '__main__':
    # tradeoff plots of NEQ
    create_tradeoff_plots(
        data_types=[NORMAL_CUSTOM, NETFLIX, MOVIE_LENS],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            NEQ_MIPS
        ],
        top_k=TRADEOFF_TOPK,
        include_error_bar=False
    )
    create_tradeoff_plots(
        data_types=[ADVERSARIAL_CUSTOM],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            NEQ_MIPS
        ],
        top_k=TRADEOFF_TOPK,
        max_speedup=MAX_SPEEDUP,
        include_error_bar=False
    )

