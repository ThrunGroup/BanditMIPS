import os
import glob
from pandas import read_csv

from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    GPT2_LM_HEAD,
    OPT_LM_HEAD,
    NETFLIX,
    MOVIE_LENS,
    ACTION_ELIMINATION,
    GREEDY_MIPS,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
    BUCKET_ACTION_ELIMINATION,
    NAIVE,
    PCA_MIPS,
    LSH_MIPS,
    H2ALSH,
)


def runtime_scaling_exp(data_type: str):
    # Hard-coded parameters
    size_minmax = (10, 1000)
    if data_type == OPT_LM_HEAD:
        epsilon = 1.0
        delta = 0.9
        maxmin = (1, -1)
        num_experiments = 10
        num_signals = 1
        size_minmax = (10, 4096)
        num_atoms = 10000
    else:
        epsilon = 0.1
        delta = 0.1
        maxmin = (5, 0)
        num_experiments = 10
        num_signals = 1
        num_atoms = 1000
        if data_type == MOVIE_LENS:
            size_minmax = (10, 6000)
        elif data_type == NETFLIX:
            size_minmax = (10, 100000)

    log_dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runtime_comparison"
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
        NAIVE,
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
            num_atoms=num_atoms,
            num_signals=num_signals,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
            num_best_atoms=1,
            num_seeds=5,
            log_dirname=log_dirname,
            data_type=data_type,
        )



def plot_runtime_scaling(is_plot_runtime: bool = False):
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        BUCKET_ACTION_ELIMINATION,
        NAIVE,
    ]
    log_dirname = "runtime_comparison"
    dir_name = os.path.join(log_dirname, "logs")
    create_scaling_plots(
        data_types=[OPT_LM_HEAD, NETFLIX, MOVIE_LENS],
        alg_names=algorithms,
        include_error_bar=False,
        dir_name=dir_name,
        is_plot_accuracy=False,
        is_logspace_x=False,
        is_fit=True,
        is_plot_runtime=is_plot_runtime,
    )
    create_scaling_plots(
        data_types=[OPT_LM_HEAD, NETFLIX, MOVIE_LENS],
        alg_names=algorithms,
        include_error_bar=False,
        dir_name=dir_name,
        is_plot_accuracy=False,
        is_logspace_x=False,
        is_fit=False,
        is_plot_runtime=is_plot_runtime,
    )


def get_speedups():
    log_dirname = "runtime_comparison"
    dir_name = os.path.join(log_dirname, "logs")
    data_types = [OPT_LM_HEAD, NETFLIX, MOVIE_LENS]
    for datatype in data_types:
        csv_files = glob.glob(
            # os.path.join(log_dir, f"*SCALING_{datatype}*ind_var{ind_variable}*")
            os.path.join(dir_name, f"*SCALING_{datatype}*")
        )

        # Naive algorithm
        for f in csv_files:
            if f.find(NAIVE) < 0:
                continue
            data = read_csv(f)
            independent_var = (
                "signal_sizes" if "signal_sizes" in data.keys() else "num_atoms"
            )

            x = data[independent_var]
            y = data["runtime"] * 1000

            # First experiment contains compiling time for numba
            x = x[1:]
            y = y[1:]
            naive_runtimes = y

        # All algorithms
        for f in csv_files:
            print(f"\n=={f}")
            data = read_csv(f)
            independent_var = (
                "signal_sizes" if "signal_sizes" in data.keys() else "num_atoms"
            )

            x = data[independent_var]
            y = data["runtime"] * 1000

            # First experiment contains compiling time for numba
            x = x[1:]
            y = y[1:]
            y = naive_runtimes / y
            x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))

            for idx in range(len(x)):
                print(f"signal size: {x[idx]} / speedup: {y[idx]}x")

def exp_runtime_scaling():
    for data_type in [OPT_LM_HEAD, NETFLIX, MOVIE_LENS]:
        runtime_scaling_exp(data_type=data_type)
    plot_runtime_scaling(is_plot_runtime=True)
    get_speedups()


if __name__ == "__main__":
    exp_runtime_scaling()