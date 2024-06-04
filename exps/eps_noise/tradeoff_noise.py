import os
from typing import List

from exps.plot_baselines import create_tradeoff_plots
from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    NETFLIX,
    LEN_SIGNAL_NETFLIX,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_EXPERIMENTS
)


def tradeoff_noise(algorithms, is_plot: bool = False):
    """
    See if the tradeoff behavior for the MIPS algorithms is consistent even for the noise-induced Netflix dataset.
    By default, the algorithm tested is BanditMIPS.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "normalized_logs", f"topk_{top_k}")
        for alg in algorithms:
            speedup_precision_exps(
                num_atoms=TRADEOFF_NUM_ATOMS,
                num_experiments=TRADEOFF_NUM_EXPERIMENTS,
                len_signals=LEN_SIGNAL_NETFLIX,
                num_signals=30,
                data_type=NETFLIX,
                add_noise=True,
                mips_alg=alg,
                with_replacement=False,
                dir_name=log_dir,
                num_best_atoms=top_k,
                num_seeds=10,
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


if __name__ == "__main__":
    tradeoff_noise([ACTION_ELIMINATION])