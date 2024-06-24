import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pandas import read_csv
from typing import List, Any
from utils.constants import (
    NUM_SEEDS,
    COLORS,
    ALG_TO_COLOR,
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    NETFLIX_TRANSPOSE,
    MOVIE_LENS,
    COR_NORMAL_CUSTOM,
    CRYPTO_PAIRS,
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,
    TOY_SINE,
    MNIST_T,
    SIMPLE_SONG,
    SIFT_1M,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    POSITIVE_COR_NORMAL_CUSTOM,
    HIGHLY_SYMMETRIC,
    GPT2_LM_HEAD,
    OPT_LM_HEAD,

    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    H2ALSH,
    HNSW_MIPS,
    NAPG_MIPS,
    NAIVE,

    SCALING_LOG_BUFFER,
    MAX_SPEEDUP,
    NEQ_MIPS,

    MAX_SPEEDUP,
    TRADEOFF_TOPK,
    MAX_SPEEDUP,
    BUCKET_ACTION_ELIMINATION,
    DIMENSION_OF_ATOMS,
    NUMBER_OF_ATOMS,
)
from utils.utils import fit_and_plot


def translate_alg_name(alg_name: str):
    """
    Translates algorithm names from the codebase into algorithm names consistent with the paper.

    :param alg_name: the algorithm name used throughout the code
    :returns: the translated algorithm name.
    """
    if alg_name == ACTION_ELIMINATION:
        return "BanditMIPS"
    elif alg_name == ADAPTIVE_ACTION_ELIMINATION:
        return "BanditMIPS-$\\alpha$"
    elif alg_name == MEDIAN_ELIMINATION:
        return "BoundedME"
    elif alg_name == GREEDY_MIPS:
        return "GREEDY-MIPS"
    elif alg_name == LSH_MIPS:
        return "LSH-MIPS"
    elif alg_name == PCA_MIPS:
        return "PCA-MIPS"
    elif alg_name == HNSW_MIPS:
        return "HNSW-MIPS"
    elif alg_name == NAPG_MIPS:
        return "NAPG-MIPS"
    elif alg_name == NEQ_MIPS:
        return "NEQ_MIPS"
    elif alg_name == BUCKET_ACTION_ELIMINATION:
        return "Bucket_AE"
    elif alg_name == H2ALSH:
        return "H2ALSH"
    elif alg_name == NAIVE:
        return "Naive_MIPS"
    else:
        raise Exception("Bad algorithm name")


def translate_dataset_name(dataset_name: str):
    """
    Translates the dataset names from the codebase into dataset names consistent with the paper.

    :param dataset_name: the dataset name used throughout the code
    :returns: the translated dataset name.
    """
    if dataset_name == ADVERSARIAL_CUSTOM:
        # return "Synthetic Adversarial"
        return ADVERSARIAL_CUSTOM
    elif dataset_name == NORMAL_CUSTOM:
        # return "Synthetic Gaussian"
        return NORMAL_CUSTOM
    elif dataset_name == NORMAL_PAPER:
        return NORMAL_PAPER
    elif dataset_name == NETFLIX:
        return "Netflix"
    elif dataset_name == NETFLIX_TRANSPOSE:
        return "Netflix_transposed"
    elif dataset_name == MOVIE_LENS:
        return "Movie Lens"
    elif dataset_name == COR_NORMAL_CUSTOM:
        return COR_NORMAL_CUSTOM
    elif dataset_name == CRYPTO_PAIRS:
        return "Crypto Pairs"
    elif dataset_name == CLEAR_LEADER_HARD:
        return "Clear Leader (Hard)"
    elif dataset_name == CLEAR_LEADER_SOFT:
        return "Clear Leader (Soft)"
    elif dataset_name == NO_CLEAR_LEADER:
        return "No Clear Leader"
    elif dataset_name == TOY_SINE:
        return "Toy Sine"
    elif dataset_name == MNIST_T:
        return "MNIST Transposed"
    elif dataset_name == SIMPLE_SONG:
        return "Simple Song"
    elif dataset_name == SIFT_1M:
        return "SIFT 1M"
    elif dataset_name == UNIFORM_PAPER:
        return "Uniform Paper"
    elif dataset_name == POSITIVE_COR_NORMAL_CUSTOM:
        return "Positive Correlated Normal Custom"
    elif dataset_name == NORMAL_PAPER:
        return "Synthetic Normal"
    elif dataset_name == HIGHLY_SYMMETRIC:
        return "Highly Symmetric Normal"
    elif dataset_name == GPT2_LM_HEAD:
        return "GPT2 Final Layer"
    elif dataset_name == OPT_LM_HEAD:
        return "OPT3 Final Layer"
    else:
        raise Exception("Bad dataset name {}".format(dataset_name))


def extract_alg(filename: str):
    """
    Extracts algorithm name from given filename. Used to search for specific algorithms in the log files.
    """
    if (
        BUCKET_ACTION_ELIMINATION in filename
    ):  # Order matters as if BUCKET_ACTION_ELIMINATION is in filename,
        # ACTION_ELIMINATION is also in filename
        return BUCKET_ACTION_ELIMINATION
    elif ADAPTIVE_ACTION_ELIMINATION in filename:
        return ADAPTIVE_ACTION_ELIMINATION
    elif ACTION_ELIMINATION in filename:
        return ACTION_ELIMINATION
    elif MEDIAN_ELIMINATION in filename:
        return MEDIAN_ELIMINATION
    elif GREEDY_MIPS in filename:
        return GREEDY_MIPS
    elif LSH_MIPS in filename:
        return LSH_MIPS
    elif PCA_MIPS in filename:
        return PCA_MIPS
    elif NAPG_MIPS in filename:
        return NAPG_MIPS
    elif HNSW_MIPS in filename:
        return HNSW_MIPS
    elif NEQ_MIPS in filename:
        return NEQ_MIPS
    elif BUCKET_ACTION_ELIMINATION in filename:
        return BUCKET_ACTION_ELIMINATION
    elif H2ALSH in filename:
        return H2ALSH
    elif NAIVE in filename:
        return NAIVE
    else:
        raise Exception("No algorithm was able to be extracted from " + filename)


def power10(array):
    return np.power(10, array)


def power10_sqrt(array):
    return np.power(10, array) ** (1 / 2)


def get_x_label(is_logspace: bool = False, independent_var: str = None):
    x_label = ""
    if independent_var == "signal_sizes":
        x_label = "Signal Vector Size ($d$)"
    else:
        x_label = "Number of atoms ($N$)"

    if is_logspace:
        x_label = "Log$_{10}$ " + x_label

    return x_label


def create_scaling_plots(
    caching_type: str = "",
    data_types: List[str] = [],
    alg_names: List[str] = [],
    include_error_bar: bool = False,
    ind_variables: List[str] = [DIMENSION_OF_ATOMS],
    is_plot_runtime: bool = False,
    is_fit: bool = False,
    is_logspace_x: bool = False,
    is_logspace_y: bool = False,
    is_plot_accuracy: bool = False,
    dir_name: str = None,
    save_to: str = None,
):
    """
    Plot the scaling experiments from the data stored in the logs file.

    :param caching_type: either "", "(naive_cache)", "(PI_cache)"
    :param data_types: the datasets that you want to plot. If empty, warning is triggered.
    :param alg_names: the algorithms that you want to plot. If empty, warning is triggered.
    :param ind_variables: Independent variables (N or D)
    :param include_error_bar: shows the standard deviation
    :param is_plot_runtime: whether to plot runtime
    :param is_fit: whether to fit the plots or not
    :param is_logspace_x: whether to plot x-axis in logspace or not
    :param is_logspace_y: whether to plot y-axis in logspace or not
    :param is_plot_accuracy: whether to annotate accuracy in the graph
    :param dir_name: directory name of log files
    """
    # get all csv files
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(parent_dir, dir_name) 
  
    if len(data_types) == 0:
        raise Exception("At least one dataset must be specified")

    if len(alg_names) == 0:
        raise Exception("At least one algorithm must be specified")


    for datatype in data_types:
        csv_files = glob.glob(os.path.join(log_dir, f"*SCALING_{datatype}*"))
        title = f"Scaling Baselines: {translate_dataset_name(datatype)}"
        for f in csv_files:
            alg_name = extract_alg(f)
            data = read_csv(f)
            independent_var = (
                "signal_sizes" if "signal_sizes" in data.keys() else "num_atoms"
            )

            # only get relevant data
            if f.find(datatype) < 0 or alg_name not in alg_names:
                continue

            x = data[independent_var].tolist()
            if is_plot_runtime:
                y = data["runtime"] * 1000
                ylabel = "Runtime (ms)"

                # First experiment contains compiling time for numba
                x = x[1:]
                y = y[1:]
                error = data["budgets_std"].tolist()[1:]
            else:
                y = data["budgets"].tolist()
                ylabel = "Sample Complexity"
                error = data["budgets_std"].tolist()
            x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))  # sort

            # if is_plot_runtime:
            #     diff = max(y) - min(y)
            #     plt.ylim(min(y) - diff / 2, max(y) + diff / 2)

            if not include_error_bar:
                error = np.zeros_like(error)

            if is_fit:
                is_logspace = (is_logspace_x and is_logspace_y)
                title = f"{translate_alg_name(alg_name)} on {translate_dataset_name(datatype)}"
                if is_logspace_x:
                    functions = [np.array, power10_sqrt, power10]
                    function_names = [
                        "$y \propto log x$",
                        "$y \propto x^{1/2}$",
                        "$y \propto x$",
                    ]
                    x = np.log(x)
                elif is_logspace_y:
                    y = np.log(y)

                else:
                    functions = [np.array, np.sqrt, np.log]
                    function_names = ["Linear fit", "Sqrt fit", "Logarithmic fit"]

                fit_and_plot(
                    xs=x,
                    ys=y,
                    function_list=functions,
                    function_name_list=function_names,
                    ys_std=error,
                    is_logspace=is_logspace_x,
                )
                plt.title(title)
                # if not is_plot_runtime:
                #     plt.ylim(3200, 4300)

                xlabel = get_x_label(is_logspace=is_logspace_x, independent_var=independent_var)
                if is_logspace_y:
                    ylabel = "$\ln" + ylabel + "$"
                plt.ylabel(ylabel)
                plt.xlabel(xlabel)
            else:
                if is_logspace_y:
                    y = np.log(y)
                    error = np.where(np.array(error) < SCALING_LOG_BUFFER, SCALING_LOG_BUFFER, np.array(error))
                    error = np.log(error) / (y * np.sqrt(NUM_SEEDS))

                plt.scatter(
                    x,
                    y,
                    color=ALG_TO_COLOR[alg_name],
                    label=f"{translate_alg_name(alg_name)}",
                )
                plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

                # Sort the legend entries (labels and handles) by labels
                handles, labels = plt.gca().get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0])
                )
                plt.legend(handles, labels, loc="upper left")

                if include_error_bar and all(e >= 0 for e in error):
                    plt.errorbar(x, y, yerr=error, fmt=".", color="black")

                plt.title(title)
                xlabel = (
                    "Signal Vector Size ($d$)"
                    if independent_var == "signal_sizes"
                    else "Number of atoms ($N$)"
                )

                if is_logspace_y:
                    ylabel = "$\ln{" + ylabel + "}$"

                plt.ylabel(ylabel)
                plt.xlabel(xlabel)
     
            if is_plot_accuracy:
                avg_accuracy = np.mean(data["accuracy"])
                plt.text(
                    x[-1],
                    y[-1],
                    f"Accuracy: {avg_accuracy}",
                    ha="right",
                    va="bottom",
                )
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, datatype))


def create_tradeoff_plots(
    data_types: List[str] = [],
    alg_names: List[str] = [],
    top_k: int = 1,
    include_error_bar: bool = False,
    max_speedup: int = None,
    log_dir: str = None,
    is_logspace: bool = True,
):
    """
    Plot the tradeoff experiments from the data stored in the logs file.

    :data_types: the datasets that you want to plot. If empty, warning is triggered.
    :alg_names: the algorithms that you want to plot. If empty, warning is triggered.
    :include_error_bar: shows the standard deviation
    :max_speedup: only displays speedup up to max_speedup
    """
    # get all csv files
    if log_dir is None:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join("tradeoff", "logs")
        log_dir = os.path.join(parent_dir, log_dir)

    if len(data_types) == 0:
        raise Exception("At least one dataset must be specified")
    if len(alg_names) == 0:
        raise Exception("At least one algorithm must be specified")

    for datatype in data_types:
        csv_files = glob.glob(
            os.path.join(log_dir, f"*SPEEDUP_{datatype}*")
        )
        plt.figure(figsize=(6, 4.5))
        title = (
            f"Precision-Speed Tradeoff Baselines: {translate_dataset_name(datatype)}"
        )
        for f in csv_files:
            alg_name = extract_alg(f)
            data = read_csv(f)

            # only get relevant data
            if (
                f.find(datatype) < 0
                or alg_name not in alg_names
                or f.find(str("topk") + str(top_k)) < 0
            ):
                continue

            budget_data = data["budgets"].to_list()
            naive_budget = data["naive_budget"].to_list()
            x = np.array(naive_budget) / np.array(budget_data)  # speed is inverse budget
            y = data["accuracy"].tolist()

            # Delete indices where speedup is less than 1
            if max_speedup is None:
                max_speedup = float('inf')
            del_idcs = np.where((x < 1) | (x > max_speedup))[0]
            x = np.delete(x, del_idcs)
            y = np.delete(y, del_idcs)

            if len(x) == 0:
                continue

            error = data["accuracy_std"].tolist()
            x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))

            # Plot (1, 1) on every plots
            x = np.concatenate(([1], x))
            y = np.concatenate(([1], y))

            # Plot the moving averages of accuracy
            window_size = min(5, len(y))  # Hard-coded
            x = np.concatenate((np.full(window_size-1, x[0]), x, np.full(window_size-1, x[-1])))
            y = np.concatenate((np.full(window_size - 1, y[0]), y, np.full(window_size - 1, y[-1])))
            y = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
            x = x[:-(window_size-1)]

            if is_logspace:
                plt.xscale("log")

            # plot the datapoints
            plt.scatter(
                x,
                y,
                color=ALG_TO_COLOR[alg_name],
                label=f"{translate_alg_name(alg_name)}",
            )
            plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

            # Sort the legend entries (labels and handles) by labels
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels, loc="lower right")

            if include_error_bar:
                plt.errorbar(x, y, error, fmt=".", color="black")

            plt.title(title)
            xlabel = "Speedup"
            ylabel = f"Precision at {top_k}"
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            # plt.ylim(0.5, 1.2)

        fig = plt.gcf()
        fig.set_size_inches(6, 4.5)
        plt.show()


def compare_caching_budgets(
    dir_name="mp_scaling",
    caching_type="",
    data_types=[NETFLIX],
    alg_names=[ACTION_ELIMINATION],
    include_error_bar=True,
):
    """
    This function is used to plot the budget costs of PI caching and naive caching side-by-side to see that PI
    caching has better computational costs.

    :param dir_name: the directory name of where the log files are stored
    :param caching_type: either PI or naive
    """
    # get all csv files
    path = os.path.join(os.getcwd(), "logs")
    path = os.path.join(path, dir_name)
    csv_files = glob.glob(os.path.join(path, "*SCALING*"))

    if len(data_types) == 0:
        raise Exception("At least one dataset must be specified")
    if len(alg_names) == 0:
        raise Exception("At least one algorithm must be specified")

    for datatype in data_types:
        plt.figure(figsize=(6, 4.5))
        title = f"Scaling Baselines: {translate_dataset_name(datatype)}"
        for f in csv_files:
            alg_name = extract_alg(f)
            data = read_csv(f)

            # only get relevant data
            if f.find(datatype) < 0 or f.find(caching_type) < 0 or alg_name not in alg_names:
                continue

            x = data["signal_sizes"].tolist()
            y = data["budgets"].tolist()
            error = data["budgets_std"].tolist()
            x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))  # sort
            plt.scatter(
                x, y,
                color=ALG_TO_COLOR[alg_name],
                label=f"{translate_alg_name(alg_name)}"
            )
            plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

            # Sort the legend entries (labels and handles) by labels
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels, loc="upper left")

            if include_error_bar:
                plt.errorbar(x, y, error, fmt=".", color="black")
            plt.title(title)
            xlabel = "Signal Vector Size (d)"
            ylabel = "Sample Complexity"
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)

        plt.show()


def generate_scaling_plots():
    """
    Generate scaling plots for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # basic scaling plots (synthetic and real datasets)
    create_scaling_plots(
        is_fit=False,
        include_error_bar=True,
        data_types=[
            # NORMAL_CUSTOM,
            # ADVERSARIAL_CUSTOM,
            NETFLIX,
            # MOVIE_LENS,
            # COR_NORMAL_CUSTOM,
        ],
        alg_names=[
            # ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            # MEDIAN_ELIMINATION,
            # GREEDY_MIPS,
            # PCA_MIPS,
            # LSH_MIPS,
            # HNSW_MIPS,
            # NAPG_MIPS,
            # LSH_MIPS,
            # NEQ_MIPS,
            # BUCKET_ACTION_ELIMINATION,
            # H2ALSH,
        ],
    )

    # # fit the bandit algorithms, and greedy-mips for better comparison
    create_scaling_plots(
        is_fit=True,
        is_logspace_x=True,
        include_error_bar=True,
        data_types=[NETFLIX, MOVIE_LENS],
        alg_names=[
            # ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            # MEDIAN_ELIMINATION,
            # GREEDY_MIPS,
        ],
    )


def generate_tradeoff_plots(top_k: int = 10):
    """
    Generate tradeoff plots for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # precision-accuracy tradeoff experiments (synthetic and real datasets)
    create_tradeoff_plots(
        data_types=[NORMAL_CUSTOM, NETFLIX, MOVIE_LENS],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
            PCA_MIPS,
            LSH_MIPS,
            BUCKET_ACTION_ELIMINATION,
            H2ALSH,
            HNSW_MIPS,
            NAPG_MIPS,
            NEQ_MIPS,
        ],
        top_k=top_k,
        include_error_bar=False,
    )
    create_tradeoff_plots(
        data_types=[ADVERSARIAL_CUSTOM],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
            PCA_MIPS,
            LSH_MIPS,
            BUCKET_ACTION_ELIMINATION,
            H2ALSH,
            NEQ_MIPS,
            HNSW_MIPS,
            NAPG_MIPS,
        ],
        top_k=top_k,
        max_speedup=MAX_SPEEDUP,
        include_error_bar=False,
    )


if __name__ == "__main__":
    generate_scaling_plots()
    # generate_tradeoff_plots()
    # generate_tradeoff_plots(top_k=5)
    # generate_tradeoff_plots(top_k=20)
