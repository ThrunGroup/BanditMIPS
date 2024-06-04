import os
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_tradeoff_plots
from utils.constants import (
    GPT2_LM_HEAD,
    ACTION_ELIMINATION,
    GREEDY_MIPS,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
    BUCKET_ACTION_ELIMINATION,
    PCA_MIPS,
    LSH_MIPS,
    H2ALSH,
)
from exps.speedup_precision_exps import speedup_precision_exps


def tradeoff_exp_gpt2():
    # Hard-coded parameters
    epsilon = 1.0
    delta = 0.9
    maxmin = (1, -1)
    num_experiments = 10
    num_signals = 10
    size_minmax = (10, 1600)
    log_dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scaling_gpt2/tradeoff_gpt2"
    )
    algorithms = [
        # PCA_MIPS,
        # LSH_MIPS,
        # H2ALSH,
        # GREEDY_MIPS,
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        BUCKET_ACTION_ELIMINATION,
        # GREEDY_MIPS,
    ]
    for alg in algorithms:
        speedup_precision_exps(
            dir_name=log_dirname,
            mips_alg=alg,
            num_seeds=3,
            num_best_atoms=1,
            num_atoms=int(1e4),
            num_signals=3,
            data_type=GPT2_LM_HEAD,
            len_signals=1600,
        )


def plot_tradeoff_gpt2():
    algorithms = [
        # PCA_MIPS,
        # LSH_MIPS,
        # H2ALSH,
        GREEDY_MIPS,
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        BUCKET_ACTION_ELIMINATION,
        # GREEDY_MIPS,
    ]
    log_dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scaling_gpt2/tradeoff_gpt2"
    )
    create_tradeoff_plots(
        data_types=[GPT2_LM_HEAD],
        alg_names=algorithms,
        log_dir=log_dirname,
    )


if __name__ ==  "__main__":
    # tradeoff_exp_gpt2()
    plot_tradeoff_gpt2()