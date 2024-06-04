import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from utils.constants import (
    CRYPTO_PAIRS,

    NUMBER_OF_ATOMS,
    DIMENSION_OF_ATOMS,

    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    PCA_MIPS,
    LSH_MIPS,
    HNSW_MIPS,
    NAPG_MIPS,
    NEQ_MIPS,
    H2ALSH,
    BUCKET_ACTION_ELIMINATION,
    NAIVE,

    NUM_SEEDS,
    SCALING_NUM_EXPERIMENTS,
    DEFAULT_MAXMIN,
    NORMALIZED_MAXMIN,
    DEFAULT_GREEDY_BUDGET,
    SCALING_SIZE_MINMAX,

    SCALING_DELTA,
    SCALING_DEPTH,
    SCALING_TOPK,
    SCALING_EPSILON,
    SCALING_NUM_HFUNC,
    SCALING_NUM_TABLES,
    SCALING_H2ALSH_DELTA,
    SCALING_H2ALSH_C0,
    SCALING_H2ALSH_C,
    SCALING_H2ALSH_N0,

    NOISE_VAR,
)

from neq_mips.constants import (
    PARTITION,
    NUM_CODEBOOKS,
    NUM_CODEWORDS,
)

from utils.utils import get_sizes_arr
from data.get_data import get_data
from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips
from algorithms.bucket_ae import bucket_action_elimination
from algorithms.h2alsh_mips import H2ALSH_MIPS

from algorithms.quantization.norm_pq import NormPQ
from algorithms.quantization.residual_pq import ResidualPQ
from algorithms.quantization.base_pq import PQ
from algorithms.quantization.utils import execute

from algorithms.napg import hnsw


def scaling_exp(
    epsilon: Optional[float] = SCALING_EPSILON,
    delta: Optional[float] = SCALING_DELTA,
    maxmin: Tuple[float, float] = DEFAULT_MAXMIN,
    data_type: str = NORMAL_CUSTOM,
    add_noise: bool = False,
    mips_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = SCALING_NUM_EXPERIMENTS,
    num_seeds: int = NUM_SEEDS,
    data_generation_seed: int = 0,
    size_minmax: Tuple[int, int] = SCALING_SIZE_MINMAX,
    independent_var: str = DIMENSION_OF_ATOMS,
    num_atoms: int = 10**3,
    num_signals: int = 1,
    signal_dim: int = 10 ** 2,
    num_best_atoms: int = SCALING_TOPK,
    with_replacement: bool = False,
    is_normalize: bool = False,
    is_log: bool = True,
    is_logspace: bool = False,
    dir_name: str = None,
    verbose: bool = False,
    log_dirname: str = "",
) -> float:
    """
    Run the time complexity of the bandit and baseline algorithms as the signal size scales and store them in
    the appropriate log files. MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in
    "https://arxiv.org/pdf/1812.06360.pdf#page=5".
    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param maxmin: Maximum and minimum of data
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param num_experiments: Number of experiments, determines the different sizes of d that we run
    :param num_seeds: Number of seeds
    :param data_generation_seed: Random seed for generating data
    :param independent_var: The variable that is not constant (Either N=number of atoms or D=Dimension of atom)
    :param size_minmax: Minimum and Maximum of independent variable
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param signal_dim: Dimension of atom/signal
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dir_name: Directory name to log the results
    :param verbose: Whether to print the number of iterations ran for action elimination
    :param log_dirname: log filename
    """
    avg_runtimes = np.zeros(
        (num_experiments, num_seeds)
    )  # Note that the runtime is average over all signals
    budgets = np.zeros((num_experiments, num_seeds, num_signals))
    accuracies = np.zeros((num_experiments, num_seeds, num_signals))

    if independent_var == DIMENSION_OF_ATOMS:
        sizes = get_sizes_arr(is_logspace, size_minmax, num_experiments=num_experiments)
        # If the dataset is a toy dataset, the randomness will include the data generation process
        # If the dataset is a real dataset, the randomness will only include which vector is chosen as signal
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=sizes[-1],
            num_signals=num_signals,
            seed=data_generation_seed,
            data_type=data_type,
            add_noise=add_noise,
        )
    elif independent_var == NUMBER_OF_ATOMS:
        sizes = get_sizes_arr(is_logspace, size_minmax, num_experiments=num_experiments)
        # If the dataset is a toy dataset, the randomness will include the data generation process
        # If the dataset is a real dataset, the randomness will only include which vector is chosen as signal
        atoms, signals = get_data(
            num_atoms=sizes[-1],
            len_signal=signal_dim,
            num_signals=num_signals,
            seed=data_generation_seed,
            data_type=data_type,
            add_noise=add_noise,
        )
    else:
        raise NotImplementedError(
            f"{independent_var} is not an implemented control variable"
        )

    for seed in range(num_seeds):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)

        # run the experiment for the different values of independent variable
        for size_idx, size in enumerate(sizes):
            subset_indices = rng.choice(sizes[-1], size, replace=False)
            if independent_var is DIMENSION_OF_ATOMS:
                atoms_subset, signals_subset = (
                    atoms[:, subset_indices],
                    signals[:, subset_indices],
                )
            elif independent_var is NUMBER_OF_ATOMS:
                atoms_subset, signals_subset = (
                    atoms[subset_indices, :],
                    signals,
                )
            else:
                raise NotImplementedError(
                    f"{independent_var} is not an implemented control variable"
                )

            if is_normalize:
                atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
                signal_norm = np.linalg.norm(signals_subset, axis=1, keepdims=True)
                atoms_subset /= atoms_norm
                signals_subset /= signal_norm
                epsilon /= atoms_norm.max() * signal_norm
                maxmin = NORMALIZED_MAXMIN  # Assume that the information on element-wise product is not given

            # get the ground truth arrays
            naive_candidates_array = (
                np.matmul(-atoms_subset, signals_subset.transpose())
                .argsort(axis=0)[:num_best_atoms]
                .transpose()
            )

            # bandit algorithms
            if mips_alg in [
                ACTION_ELIMINATION,
                ADAPTIVE_ACTION_ELIMINATION,
                MEDIAN_ELIMINATION,
                BUCKET_ACTION_ELIMINATION,
            ]:
                log_info = (
                    "ind_var"
                    + str(independent_var)
                    + "_epsilon"
                    + str(epsilon)
                    + "_delta"
                    + str(delta)
                )
                start_time = time.time()
                bandit_results = mips_bandit(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    maxmin=maxmin,
                    bandit_alg=mips_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_atoms=num_best_atoms,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                    verbose=verbose,
                    var_proxy_override=(data_type == CRYPTO_PAIRS),
                )
                candidates_array, budget_array = bandit_results[
                    :2
                ]  # mips_bandit returns five values
                runtime = time.time() - start_time
            elif mips_alg is GREEDY_MIPS:
                log_info = (
                    "ind_var"
                    + str(independent_var)
                    + "budget"
                    + str(DEFAULT_GREEDY_BUDGET)
                )
                start_time = time.time()
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    budget=DEFAULT_GREEDY_BUDGET,
                    num_best_atoms=num_best_atoms,
                )
                runtime = time.time() - start_time
            elif mips_alg is PCA_MIPS:
                log_info = "ind_var" + str(independent_var) + "depth" + str(delta)
                start_time = time.time()
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    delta=SCALING_DEPTH,
                )
                runtime = time.time() - start_time
            elif mips_alg is LSH_MIPS:
                log_info = (
                    "ind_var"
                    + str(independent_var)
                    + "hfunc_"
                    + str(SCALING_NUM_HFUNC)
                    + "_tables"
                    + str(SCALING_NUM_TABLES)
                )
                start_time = time.time()
                candidates_array, budget_array = run_lsh_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    num_hfunc=SCALING_NUM_HFUNC,
                    num_tables=SCALING_NUM_TABLES,
                )
                runtime = time.time() - start_time
            elif mips_alg in [HNSW_MIPS, NAPG_MIPS]:
                log_info = "ind_var" + str(independent_var)
                start_time = time.time()
                candidates_array, budget_array = hnsw(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    use_norm_adjusted_factors=(mips_alg == NAPG_MIPS),
                )
                runtime = time.time() - start_time
            elif mips_alg is NEQ_MIPS:
                log_info = (
                    "_codebooks"
                    + str(NUM_CODEBOOKS)
                    + "_codewords"
                    + str(NUM_CODEWORDS)
                )
                pqs = [
                    PQ(M=PARTITION, Ks=NUM_CODEWORDS) for _ in range(NUM_CODEBOOKS - 1)
                ]
                quantizer = ResidualPQ(pqs=pqs)
                quantizer = NormPQ(n_percentile=NUM_CODEWORDS, quantize=quantizer)
                start_time = time.time()

                candidates_array, budget_array = execute(
                    seed=seed,
                    top_k=num_best_atoms,
                    pq=quantizer,
                    X=atoms_subset,
                    Q=signals_subset.astype("float32"),
                    G=naive_candidates_array,
                    num_codebooks=NUM_CODEBOOKS,
                    num_codewords=NUM_CODEWORDS,
                    train_size=num_atoms,
                )
                # includes the time finding recall (other baselines don't)
                runtime = time.time() - start_time
            elif mips_alg is H2ALSH:
                log_info = (
                    "_delta"
                    + str(SCALING_H2ALSH_DELTA)
                    + "_c0"
                    + str(SCALING_H2ALSH_C0)
                    + "_c"
                    + str(SCALING_H2ALSH_C)
                    + "_N0"
                    + str(SCALING_H2ALSH_N0)
                )
                h2alsh = H2ALSH_MIPS(
                    atoms=atoms_subset,
                    delta=SCALING_H2ALSH_DELTA,
                    c0=SCALING_H2ALSH_C0,
                    c=SCALING_H2ALSH_C,
                    N0=SCALING_H2ALSH_N0,
                )
                start_time = time.time()
                candidates_array, budget_array = h2alsh.mip_search_queries(
                    queries=signals_subset, top_k=num_best_atoms
                )
                runtime = time.time() - start_time
            elif mips_alg is NAIVE:
                start_time = time.time()
                candidates_array = (
                    np.matmul(-atoms_subset, signals_subset.transpose())
                    .argsort(axis=0)[:num_best_atoms]
                    .transpose()
                )
                budget_array = len(signals_subset) * len(signals_subset[0])
                runtime = time.time() - start_time
                log_info = ""
            else:
                raise NotImplementedError(f"{mips_alg} is not implemented")

            # store runtime, budget, budget_std, and accuracy of the experiment
            avg_runtimes[size_idx, seed] = (
                runtime / num_signals
            )  # Note that the runtimes are averaged over all signals

            # assert budget_array.size == num_signals, "budget_array should be the size of the signals"
            budgets[size_idx, seed, :] = budget_array

            naive_candidates_array = (
                np.matmul(-atoms_subset, signals_subset.transpose())
                .argsort(axis=0)[:num_best_atoms]
                .transpose()
            )
            for signal_idx in range(num_signals):
                matches = len(
                    np.intersect1d(
                        naive_candidates_array[signal_idx],
                        candidates_array[signal_idx],
                    )
                )
                accuracies[size_idx, seed, signal_idx] = matches / num_best_atoms

    avg_runtime = np.mean(
        avg_runtimes, axis=1
    )  # Average over seeds because it's already averaged over signals
    avg_budget = np.mean(budgets, axis=(1, 2))  # Average over seeds and signals
    avg_budget_std = np.std(
        budgets, axis=(1, 2)
    ) / np.sqrt(num_signals * num_seeds)  # Std over seeds and signals for given size
    avg_accuracy = np.mean(accuracies, axis=(1, 2))  # Average over seeds and signals

    assert (
        avg_runtime.size
        == avg_budget.size
        == avg_budget_std.size
        == avg_accuracy.size
        == num_experiments
    ), "The sizes of the arrays should be the same as num_experiments"

    # save to log files
    if is_log:
        if independent_var is DIMENSION_OF_ATOMS:
            log_dict = {
                "signal_sizes": sizes,
                "budgets": avg_budget,
                "budgets_std": avg_budget_std,
                "accuracy": avg_accuracy,
                "runtime": avg_runtime,
            }
        elif independent_var is NUMBER_OF_ATOMS:
            log_dict = {
                "num_atoms": sizes,
                "budgets": avg_budget,
                "budgets_std": avg_budget_std,
                "accuracy": avg_accuracy,
                "runtime": avg_runtime,
            }
        else:
            raise NotImplementedError(
                f"{independent_var} is not an implemented control variable"
            )

        log_df = pd.DataFrame(log_dict)
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        if add_noise:
            experiment = "eps_noise"
            noise_factor = str(NOISE_VAR)
        elif log_dirname != "":
            experiment = log_dirname
            noise_factor = ""
        else:
            experiment = "core_scaling"
            noise_factor = ""

        log_dir = (
            os.path.join(parent_dir, experiment, "logs")
            if dir_name is None
            else dir_name
        )
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            + "SCALING_"
            + data_type
            + "_"
            + "atoms"
            + str(num_atoms)
            + "_"
            + log_info
            + "_noise"
            + noise_factor
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)

    print(f"Mean accuracy of {mips_alg} for {data_type}: {np.mean(avg_accuracy)} \n")
    return np.mean(avg_accuracy)