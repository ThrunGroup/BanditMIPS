import os
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import HIGHLY_SYMMETRIC, ACTION_ELIMINATION


def eps_suboptimal_exp():
    # Hard-coded parmeters
    epsilon = 0.5
    delta = 0.5
    maxmin = (1, -1)
    num_experiments = 15
    num_signals = 3
    size_minmax = (10**3, 10**5)
    mips_alg = ACTION_ELIMINATION
    log_dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epsilon_suboptimal")

    scaling_exp(
        epsilon=epsilon,
        delta=delta,
        maxmin=maxmin,
        size_minmax=size_minmax,
        num_experiments=num_experiments,
        mips_alg=mips_alg,
        num_signals=num_signals,
        with_replacement=False,
        is_normalize=False,
        is_logspace=False,
        num_best_atoms=1,
        num_seeds=5,
        log_dirname=log_dirname,
        data_type=HIGHLY_SYMMETRIC,
    )


def plot_eps_suboptimal():
    log_dirname = "epsilon_suboptimal"
    dir_name = os.path.join(
        log_dirname, "logs"
    )
    print(dir_name)
    create_scaling_plots(
        data_types=[HIGHLY_SYMMETRIC],
        alg_names=[ACTION_ELIMINATION],
        include_error_bar=True,
        dir_name=dir_name,
        is_fit=True,
    )


if __name__ == "__main__":
    eps_suboptimal_exp()
    plot_eps_suboptimal()
