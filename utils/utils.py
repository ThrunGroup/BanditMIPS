import os
import argparse
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from typing import Tuple, List, Any
from sklearn.linear_model import LinearRegression
from pandas import read_csv
from collections import defaultdict
from utils.constants import (
    COLORS,
    SCALING_SIZE_MINMAX,
    SCALING_NUM_EXPERIMENTS,
    CACHING_BUDGET,
    NOISE_VAR
)



@nb.njit
def set_seed(a: int):
    """
    We need this function to properly set random seeds when using numba.
    It is INCORRECT to call np.random.seed directly in numba-decorated functions!
    See shorturl.at/gPVXZ.

    :param a: seed value
    :return: None
    """
    np.random.seed(a)


def get_args() -> argparse.Namespace:
    """
    Gather the experimental arguments from the command line.

    :return: Namespace object from argparse containing each argument
    """
    parser = argparse.ArgumentParser(description="Solve matching pursuit problems")
    parser.add_argument("-N", help="Number of atoms to use", default=100, type=int)
    parser.add_argument("-d", help="Dimensionality of data", default=1000, type=int)
    parser.add_argument(
        "--seed", help="Random seed for experiments", default=1, type=int
    )
    parser.add_argument(
        "--p",
        help="Confidence parameter that governs the width of the confidence intervals",
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of coordinates to sample at once",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--max_iter", help="Maximum number of atoms to identify", default=10, type=int
    )
    parser.add_argument(
        "--tol",
        help="Minimum absolute value of inner product between atom and signal",
        default=1e-2,
        type=float,
    )
    parser.add_argument(
        "--residue",
        help="Threshold for norm of residue by which to terminate iterative algorithm.",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "-e",
        "--exp_config",
        help="Experiment configuration file to use",
        required=False,
    )
    parser.add_argument(
        "-f", "--force", help="Recompute Experiments", action="store_true"
    )
    return parser.parse_args()


def remap_args(args: argparse.Namespace, exp: List):
    """
    Parses an experiment config line (as a list) into an args variable (a Namespace).

    :param args: Namespace object whose args are to be modified
    :param exp: Experiment configuration (list of arguments)
    """
    args.N = exp[1]
    args.d = exp[2]
    args.seed = exp[3]
    args.p = None  # To compute on the fly
    args.batch_size = exp[5]
    args.max_iter = exp[6]
    args.tol = exp[7]
    args.residue = exp[8]
    return args


def q(str_: str) -> str:
    """
    Helper function that prepends and appends double quotes to a string.

    :param str_: string to put double quotes around
    """
    return '"' + str_ + '"'


def b(num: Any) -> str:
    """
    Helper function that wraps brackets to a number/string.

    :param num: number to put brackets around
    """
    return "{" + str(num) + "}"


def r(num: int):
    return round(num, 3)


def write_exp(algo, N, d, seed, p, batch_size, max_iter, tol, residue) -> str:
    """
    Given a set of experimental arguments, produce a string representing the list of arguments (to be stored in a text
    file)

    :param algo: algorithm to use, e.g., MP BanditMP
    :param N: dataset size
    :param d: dimensionality
    :param seed: random seed
    :param p: confidence parameter
    :param batch_size: batch size to use when sampling
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :param residue: residue minimum
    :return: string representation of list of arguments
    """
    return (
        "\t["
        + ", ".join(
            map(str, [q(algo), N, d, seed, p, batch_size, max_iter, tol, residue])
        )
        + "],\n"
    )


def conv_to_str(input_: List[int]) -> str:
    """
    Convenience function for printing an array of numbers to the same number of decimal places. Return a string
    representation of the given input array for easy printing.
    :param input_: input array to convert to a string
    :return: string representation of the array
    """
    return ", ".join(map(str, map(lambda num: np.round(num, decimals=5), input_)))


def write_log(
    args: argparse.Namespace,
    stats: List,  # Actually a List[List[np.ndarray, int, int, float]]
    coefs: np.ndarray,
    init_signal_norm: float,
    final_signal_norm: float,
    total_cost: int,
    bandit: bool,
):
    """
    Writes a logfile containing information about the experiment. The logfile contains information that is very useful
    for debugging, such as exactly which atoms were chosen at each step and the total cost of the algorithm.

    :param args: arguments about the experiment.
    :param stats: statistics about the experiments
    :param coefs: coefficients (inner products) with each atom at each iteration of MP
    :param init_signal_norm: initial norm of the input signal
    :param final_signal_norm: final norm of the residual (should be very small) - useful for debugging
    :param total_cost: cost of the experiment, measured in number of coordinate-wise multiplications performed
    :param bandit: whether the experiment used BanditMP (True) or MP (False)
    :return: None
    """
    nonzero_coefs = list(np.where(np.abs(coefs) > 0)[0])
    nonzero_weights = list(coefs[nonzero_coefs])
    with open(get_logfile_name(args, bandit), "w+") as fout:
        fout.write("Total cost: " + str(total_cost) + "\n")
        fout.write("Final answer:\n")
        fout.write("Nonzero indcs: " + conv_to_str(nonzero_coefs) + "\n")
        fout.write("Nonzero coefs: " + conv_to_str(nonzero_weights) + "\n")
        fout.write("Initial signal norm: " + str(init_signal_norm) + "\n")
        for row_i, row in enumerate(stats):
            fout.write("Iteration: " + str(row_i + 1) + "\n")
            fout.write(conv_to_str(row))
            fout.write("\n")
        fout.write("Final signal norm: " + str(final_signal_norm) + "\n")


def get_logfile_name(args: argparse.Namespace, bandit=True) -> str:
    """
    Returns the name of the logfile with the given arguments.

    :param args: arguments of the experiment
    :param bandit: whether to use BanditMP (True) or MP (False)
    :return: string representing the logfile's name
    """
    suffix = "-bmp" if bandit else ""
    return os.path.join(
        "../logs",
        "N-"
        + str(args.N)
        + "-d-"
        + str(args.d)
        + "-s-"
        + str(args.seed)
        + "-p-"
        + str(args.p)
        + "-b-"
        + str(args.batch_size)
        + "-m-"
        + str(args.max_iter)
        + "-r-"
        + str(args.residue)
        + "-t-"
        + str(args.tol)
        + suffix,
    )


def welford_variance_calc(
    n1: int, mean1: float, var1: float, n2: int, mean2: float, var2: float
) -> float:
    """
    Calculate the variance of A U B given |A|, |B|, mean(A), mean(B), var(A), and var(B). See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    if var1 == 0:
        return var2
    elif var2 == 0:
        return var1

    n = n1 + n2
    delta = mean1 - mean2
    M1 = n1 * var1
    M2 = n2 * var2
    M = M1 + M2 + (delta**2) * (n1 * n2) / n
    return M / n


def vectorize_welford(
    n1_vec: np.ndarray,
    mean1_vec: np.ndarray,
    var1_vec: np.ndarray,
    n2_vec: np.ndarray,
    mean2_vec: np.ndarray,
    var2_vec: np.ndarray,
) -> np.ndarray:
    """
    A vectorized(1-dimensional) version of the function "welford_veriance_calc"
    """
    n_vec = n1_vec + n2_vec
    delta_vec = mean1_vec - mean2_vec
    M1_vec = n1_vec * var1_vec
    M2_vec = n2_vec * var2_vec
    var_vec = M1_vec + M2_vec + (delta_vec**2) * (n1_vec * n2_vec) / (n_vec**2)
    return np.nan_to_num(var_vec, posinf=0, neginf=0)


def get_wo_variance(elems: np.ndarray, pop_size: int) -> np.ndarray:
    """
    Find sample variance of elements that are sample without replacement
    """
    FPC = 1 - len(elems) / pop_size  # Finite Population Correction constant
    return np.var(elems) / len(elems) * FPC


@nb.njit
def subset_2d(
    source_array: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
):
    """
    Why to implement this: there's no faster NumPy function that can replace this.
    """
    subset_array = np.empty((len(rows), len(columns)))
    for i in range(len(rows)):
        for j in range(len(columns)):
            subset_array[(i, j)] = source_array[(rows[i], columns[j])]
    return subset_array


@nb.njit
def subset_2d_cached(
    source_array: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    use_cache: bool,
    cache: np.ndarray,
    cache_tracker: np.ndarray,
    cache_map: nb.typed.List,
):
    """
    Implements subset_2d with cache support. Assumes budget to read/write cache is the same.
    Returns the subset_array, updated cache, cache_tracker, and budget.
    """
    budget = 0
    hits = 0
    subset_array = np.empty((len(rows), len(columns)))
    for i in range(len(rows)):
        for j in range(len(columns)):
            atom_idx = rows[i]
            sample_idx = columns[j]

            # Check if sample already exists in cache.
            if use_cache and cache_map[atom_idx].get(sample_idx, -1) >= 0:
                subset_array[(i, j)] = cache[(atom_idx, cache_map[atom_idx][sample_idx])]
                budget += CACHING_BUDGET
                hits += 1
            else:
                # write to cache if space exists
                source = source_array[atom_idx, sample_idx]
                if use_cache and (cache_tracker[atom_idx] < cache.shape[1]):
                    cache[(atom_idx, cache_tracker[atom_idx])] = source
                    cache_map[atom_idx][sample_idx] = cache_tracker[atom_idx]
                    cache_tracker[atom_idx] += 1
                    budget += CACHING_BUDGET

                subset_array[(i, j)] = source
                budget += 1

    return subset_array, budget, cache, cache_tracker, cache_map, hits


def fit_and_plot(
    xs: np.ndarray,
    ys: np.ndarray,
    function_list: List[np.ufunc] = None,
    function_name_list: List[str] = None,
    ys_std: np.ndarray = None,
    is_logspace: bool = False,
) -> None:
    """
    Fit xs, ys to function and plot the points and curve.

    :param xs: Input array
    :param ys: Output array
    :param function_list: List of functions
    :param function_name_list: List of function name
    :param ys_std: Standard Deviation of Output array (used for error bar plot)
    """
    plt.figure(figsize=(6, 4.5))
    plt.ticklabel_format(style="sci", axis='y', scilimits=(0, 0))
    plt.ticklabel_format(style="sci", axis='x', scilimits=(0, 0))

    lr = LinearRegression()
    hashes = np.linspace(xs[0], xs[-1], 5000)

    if (function_list is not None) and (function_name_list is not None):
        for idx in range(len(function_list)):
            function = function_list[idx]
            function_name = function_name_list[idx]
            lr.fit(function(xs).reshape(-1, 1), ys)  # Fit xs, ys into function

            xs_predict = lr.predict(function(xs).reshape(-1, 1))
            hashes_predict = lr.predict(function(hashes).reshape(-1, 1))
            mean_ys = np.mean(ys)
            if ((ys - mean_ys) ** 2).sum() < 1e-10:
                # Gracefully handle the case where ys is a constant
                r2 = 1
            else:
                r2 = 1 - ((xs_predict - ys) ** 2).sum() / ((ys - mean_ys) ** 2).sum()
            plt.plot(
                hashes,
                hashes_predict,
                color=COLORS[idx],
                label=f"{function_name},  $R^2 = {r2:.4f}$",
            )
            plt.legend(loc="upper left")
    if ys_std is None:
        plt.scatter(xs, ys, color="black")
    else:
        plt.errorbar(xs, ys, ys_std, fmt="o", color="black")


def get_sizes_arr(
    is_logspace: bool = True,
    size_minmax: Tuple[int, int] = SCALING_SIZE_MINMAX,
    num_experiments: int = SCALING_NUM_EXPERIMENTS,
) -> np.ndarray:
    """
    Create an array of signal sizes that corresponds to the dimension "d".

    :is_logspace: determines whether "d" should be mapped logarithmically
    :size_minmax: the mininum and maximum of dimension "d".
    :num_experiments: corresponds to how many points you want to plot
    """
    if is_logspace:
        sizes_signal = np.logspace(
            np.log(size_minmax[0]), np.log(size_minmax[1]), num_experiments, base=np.e
        ).astype(np.int64)
    else:
        sizes_signal = np.linspace(
            size_minmax[0], size_minmax[1], num_experiments
        ).astype(np.int64)
    return sizes_signal


def get_recon_error(
    coeffs: np.ndarray,
    candidates: np.ndarray,
    signals: np.ndarray,
 ) -> float:
    """
    Since the signal is decomposed into a linear combination of the selected indices, we also need to check that the
    coefficients that OMP finds are accurate. This function computes the reconstruction error to give info on that.
    """
    assert len(coeffs) == candidates.shape[0], "all candidates must have corresponding coefficients"
    assert candidates.shape[1] == signals.shape[0], "candidates and signals must have same number of samples"
    return np.sum(
        np.abs(
            (signals - np.sum(coeffs * candidates, axis=0))
        ) / np.sum(np.abs(signals))
    )


def add_eps_noise(atoms, max=2.0, p=NOISE_VAR):
    """
    Given a dictionary of atoms, add max * p noise to each atom.
    This is to create dictionary of atoms to emulate epsilon noise in real datasets
    """
    n, d = atoms.shape
    # max = np.max(atoms)
    return atoms + np.random.normal(0, max * p, size=(n, d))