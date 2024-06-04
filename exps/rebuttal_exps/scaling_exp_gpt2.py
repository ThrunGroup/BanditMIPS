import os
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    GPT2_LM_HEAD,
    OPT_LM_HEAD,
    ACTION_ELIMINATION,
    GREEDY_MIPS,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
    BUCKET_ACTION_ELIMINATION,
    PCA_MIPS,
    LSH_MIPS,
    H2ALSH,
)


def scaling_exp_gpt2():
    # Hard-coded parameters
    epsilon = 1.0
    delta = 0.9
    maxmin = (1, -1)
    num_experiments = 10
    num_signals = 5
    size_minmax = (10, 4096)
    log_dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scaling_gpt2"
    )
    algorithms = [
        # PCA_MIPS,
        # LSH_MIPS,
        # H2ALSH,
        # GREEDY_MIPS,
        # ACTION_ELIMINATION,
        # ADAPTIVE_ACTION_ELIMINATION,
        # MEDIAN_ELIMINATION,
        BUCKET_ACTION_ELIMINATION,
        # GREEDY_MIPS,
    ]

    for alg in algorithms:
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            maxmin=maxmin,
            size_minmax=size_minmax,
            num_experiments=num_experiments,
            mips_alg=alg,
            num_atoms=10000,
            num_signals=num_signals,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            num_best_atoms=1,
            num_seeds=5,
            log_dirname=log_dirname,
            data_type=OPT_LM_HEAD,
        )


# def tradeoff_exp_gpt2():


def plot_scaling_gpt2(is_plot_runtime: bool = False):
    algorithms = [
        GREEDY_MIPS,
        LSH_MIPS,
        H2ALSH,
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        BUCKET_ACTION_ELIMINATION,
    ]
    log_dirname = "scaling_gpt2"
    dir_name = os.path.join(log_dirname, "logs")
    print(dir_name)
    create_scaling_plots(
        data_types=[OPT_LM_HEAD],
        alg_names=algorithms,
        include_error_bar=True,
        dir_name=dir_name,
        is_plot_accuracy=False,
        is_logspace_x=False,
        is_fit=True,
        is_plot_runtime=is_plot_runtime,
    )
    create_scaling_plots(
        data_types=[OPT_LM_HEAD],
        alg_names=algorithms,
        include_error_bar=True,
        dir_name=dir_name,
        is_plot_accuracy=False,
        is_logspace_x=False,
        is_fit=False,
        is_plot_runtime=is_plot_runtime,
    )


if __name__ == "__main__":
    # scaling_exp_gpt2()
    plot_scaling_gpt2(is_plot_runtime=True)
