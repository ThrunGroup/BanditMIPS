import os

from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    CRYPTO_PAIRS_MAXMIN,

    # dataset
    CRYPTO_PAIRS,

    # algorithms
    ADAPTIVE_ACTION_ELIMINATION,
    ACTION_ELIMINATION,
    GREEDY_MIPS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
    SCALING_NUM_ATOMS_CRYPTO_PAIRS,
    SCALING_EPSILON,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_CRYPTO_PAIRS,
)


def run_crypto_pairs_scaling(run: bool = True, plot: bool = True):
    """
    Run scaling experiment for the crypto pairs dataset

    Expected results:
    1. Action elimination finds the best atom in the first iteration
    2. Budgets remain the same (3000) regardless of the signal vector size
    """
    delta = 0.1
    algorithms = [ACTION_ELIMINATION]

    dirname = "exps/crypto_pairs/crypto_pairs_logs"
    if run:
        if os.path.exists(dirname):
            print("=> crypto pairs logs exists!")

        else: 
            for algorithm in algorithms:
                scaling_exp(
                    mips_alg=algorithm,
                    epsilon=SCALING_EPSILON,
                    delta=delta,
                    num_atoms=SCALING_NUM_ATOMS_CRYPTO_PAIRS,
                    size_minmax=SCAlING_SIZE_MINMAX_CRYPTO_PAIRS,
                    maxmin=CRYPTO_PAIRS_MAXMIN,
                    num_experiments=SCALING_NUM_EXPERIMENTS,
                    num_signals=SCALING_NUM_SIGNALS,
                    data_type=CRYPTO_PAIRS,
                    with_replacement=False,
                    is_normalize=False,
                    is_logspace=True,
                    verbose=False,
                    dir_name=dirname,
                )

    if plot:
        # fit the bandit algorithm and greedy-mips for better comparison
        create_scaling_plots(
            is_fit=True,
            is_logspace_x=False,
            is_logspace_y=False,
            include_error_bar=True,
            is_plot_accuracy=True,
            data_types=[CRYPTO_PAIRS],
            dir_name=dirname,
            alg_names=[ACTION_ELIMINATION],
        )


if __name__ == "__main__":
    run_crypto_pairs_scaling(run=False)
