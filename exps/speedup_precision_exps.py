import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from utils.constants import (
    BATCH_SIZE,
    HOEFFDING,
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    HOEFFDING,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    NEQ_MIPS,
    SYMMETRIC,
    ASYMMETRIC,
    H2ALSH,
    BUCKET_ACTION_ELIMINATION,
    HNSW_MIPS,
    NAPG_MIPS,

    # constants
    NUM_SEEDS,
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_DIMENSION,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_TOPK,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_DEPTH_MINMAX,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_HFUNC_MINMAX,
    TRADEOFF_TABLES_MINMAX,
    TRADEOFF_DEFAULT_EPSILON,
    NUMBER_OF_ATOMS,
    DIMENSION_OF_ATOMS,
    TRADEOFF_H2ALSH,
    TRADEOFF_BUCKET_ACTION_ELIMINATION,
    GREEDY_BUDGET_MINMAX,
    NORMAL_VAR,
    TRADEOFF_MAXMIN,
    DELTA_MINMAX,
    DELTA_MEDIAN,
    EPSILON_MINMAX_ACTION,
    DELTA_BINS,
    DELTA_BINS_REAL,
    DELTA_BINS_ACTION,
    EPSILON_MINMAX_MEDIAN,
    GREEDY_BUDGET_BY,
    NORMALIZED_MAXMIN,

    TRADEOFF_DELTA_MINMAX_REAL,
    TRADEOFF_MAXMIN_REAL,
    TRADEOFF_VAR_REAL,
    TRADEOFF_DEPTH_MINMAX_REAL,
    TRADEOFF_BATCH_SIZE_REAL,
    TRADEOFF_MAXMIN_CRYPTO_PAIRS,
    DELTA_POWER_MINMAX,
    DELTA_BINS_AD,
    MAXMIN_AD,
    TRADEOFF_CODEBOOK_MINMAX,
    TRADEOFF_CODEWORD_MINMAX,

    NOISE_VAR,
)
from neq_mips.constants import (
    PARTITION,
    TRADEOFF_NEQ_NUM_EXPERIMENTS,
)

from data.get_data import get_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips, generate_conditer
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips
from algorithms.h2alsh_mips import H2ALSH_MIPS
from algorithms.quantization.base_pq import PQ
from algorithms.quantization.residual_pq import ResidualPQ
from algorithms.quantization.norm_pq import NormPQ
from algorithms.quantization.utils import execute
from algorithms.napg import hnsw


def get_accuracy_array(
    true_candidates: np.ndarray,
    mips_candidates: np.ndarray,
    num_signals: int = TRADEOFF_NUM_SIGNALS,
) -> np.ndarray:
    """
    Returns the average accuracy of the candidates across the signals.
    Note that this is equivalent to recall when the topk atoms is fixed.
    We return an array of accuracies (one for each signal) so that we can compute both the mean and std.

    :param true_candidates: the candidates retrieved naively
    :param mip_candidates: the candidates returned by the mips algorithm
    :param num_signals: number of signals
    :return: an array of accuracies
    """
    accuracy = []
    for idx in range(num_signals):
        accuracy.append(len(np.intersect1d(true_candidates[idx], mips_candidates[idx])))
    return np.array(accuracy) / len(true_candidates[0])


def speedup_precision_exps(
    epsilon: Optional[float] = TRADEOFF_DEFAULT_EPSILON,
    data_type: str = NORMAL_CUSTOM,
    add_noise: bool = False,
    mips_alg: str = MEDIAN_ELIMINATION,
    len_signals: int = TRADEOFF_DIMENSION,
    num_atoms: int = TRADEOFF_NUM_ATOMS,
    num_experiments: int = TRADEOFF_NUM_EXPERIMENTS,
    num_signals: int = TRADEOFF_NUM_SIGNALS,
    num_best_atoms: int = TRADEOFF_TOPK,
    num_seeds: int = NUM_SEEDS,
    with_replacement: bool = False,
    is_log: bool = True,
    is_plot: bool = False,
    dir_name: str = None,
    lsh_type: str = ASYMMETRIC,
):
    """
    Run the precision-speed tradeoff of the bandit and baseline algorithms and store them in the appropriate
    log files. Note that what we're tuning "delta" for the bandit algorithms, "budget" for greedy-mips, "depth"
    of the pca-tree for pca-mips, "(num_hfunc, num_tables)" for lsh-mips, and "(num_codebook, num_codewords)" for
    vector quantization.

    :param epsilon: A epsilon (an additive error allowed)
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param len_signals: The dimension "d" of the signals
    :param num_atoms: Number of atoms
    :param num_experiments: Number of experiments
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param dir_name: Directory name to log the results
    :param num_seeds: Number of random seeds
    """
    avg_runtime = np.zeros(num_experiments)
    avg_budget = np.zeros(num_experiments)
    avg_accuracy = np.zeros(num_experiments)
    avg_accuracy_std = np.zeros(num_experiments)
    print(f"=> Tradeoff experiment for {mips_alg} with {data_type} dataset")

    # setting up params
    if data_type in [NETFLIX, MOVIE_LENS]:
        tradeoff_delta_minmax = TRADEOFF_DELTA_MINMAX_REAL
        tradeoff_delta_bins = DELTA_BINS_REAL
        tradeoff_maxmin = TRADEOFF_MAXMIN_REAL
        tradeoff_var = TRADEOFF_VAR_REAL
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX_REAL
        tradeoff_batch_size = TRADEOFF_BATCH_SIZE_REAL
        greedy_budget = num_atoms - 1
    elif data_type is CRYPTO_PAIRS:
        tradeoff_delta_minmax = TRADEOFF_DELTA_MINMAX_REAL
        tradeoff_delta_bins = DELTA_BINS_REAL
        tradeoff_maxmin = TRADEOFF_MAXMIN_CRYPTO_PAIRS
        tradeoff_var = 10 ** 2
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX_REAL
        tradeoff_batch_size = TRADEOFF_BATCH_SIZE_REAL
        greedy_budget = num_atoms - 1
    elif data_type is ADVERSARIAL_CUSTOM:
        tradeoff_delta_minmax = TRADEOFF_DELTA_MINMAX_REAL
        tradeoff_delta_bins = DELTA_BINS_AD
        tradeoff_maxmin = MAXMIN_AD
        tradeoff_var = TRADEOFF_VAR_REAL
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX
        tradeoff_batch_size = TRADEOFF_BATCH_SIZE_REAL
        greedy_budget = num_atoms - 1
    else:
        tradeoff_delta_minmax = DELTA_MINMAX
        tradeoff_delta_bins = DELTA_BINS
        tradeoff_maxmin = TRADEOFF_MAXMIN
        tradeoff_var = NORMAL_VAR
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX
        tradeoff_batch_size = BATCH_SIZE
        greedy_budget = num_atoms // GREEDY_BUDGET_BY

    atoms, signals = get_data(
        num_atoms=num_atoms,
        len_signal=len_signals,
        num_signals=num_signals,
        seed=0,
        data_type=data_type,
        add_noise=add_noise,
        is_tradeoff=True,
        num_best_atoms=num_best_atoms,
    )
    var_proxy = np.var(atoms[:, :1000] * signals[1, :1000])
    for seed in range(num_seeds):
        print(f"... running seed {seed}")

        # these are the oracle MIPS atoms
        naive_candidates_array = (
            np.matmul(-atoms, signals.transpose())
            .argsort(axis=0)[:num_best_atoms]
            .transpose()
        )

        runtimes = []
        budgets = []
        accuracies = []
        accuracies_std = []

        if mips_alg in [ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION, BUCKET_ACTION_ELIMINATION, MEDIAN_ELIMINATION]:
            delta1, delta2 = TRADEOFF_BUCKET_ACTION_ELIMINATION["deltas"]
            epsilon1, epsilon2 = TRADEOFF_BUCKET_ACTION_ELIMINATION["epsilons"]
            (
                bucket_num_samples1,
                bucket_num_samples2,
            ) = TRADEOFF_BUCKET_ACTION_ELIMINATION["bucket_num_samples"]
            deltas = np.geomspace(delta1, delta2, num_experiments, True)
            epsilons = np.geomspace(epsilon1, epsilon2, num_experiments, True)
            bucket_num_samples_list = np.geomspace(
                bucket_num_samples1, bucket_num_samples2, num_experiments, True
            )
            for idx in range(num_experiments):
                delta = deltas[idx]
                epsilon = epsilons[idx]
                bucket_num_samples = int(bucket_num_samples_list[idx])
                log_info = "epsilon" + str(epsilon) + "_delta" + str(delta)
                start_time = time.time()
                candidates_array, budget_array = mips_bandit(
                    atoms=atoms,
                    signals=signals,
                    maxmin=(0, np.sqrt(4 *var_proxy)),
                    bandit_alg=mips_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_atoms=num_best_atoms,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                    bucket_num_samples=bucket_num_samples,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array, num_signals
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is GREEDY_MIPS:
            conditer = generate_conditer(atoms)

            for budget in np.geomspace(2, greedy_budget, num_experiments):
                log_info = "budget" + str(budget)
                start_time = time.time()
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms,
                    signals=signals,
                    budget=int(budget),
                    num_best_atoms=num_best_atoms,
                    conditer=conditer,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array, num_signals
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is PCA_MIPS:
            for depth in np.linspace(
                tradeoff_depth_minmax[0], tradeoff_depth_minmax[1], num_experiments
            ).astype(np.int64):
                log_info = "depth" + str(depth)
                start_time = time.time()
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms,
                    signals=signals,
                    num_best_atoms=num_best_atoms,
                    delta=depth,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array, num_signals
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is LSH_MIPS:
            num_hfunc_minmax = TRADEOFF_HFUNC_MINMAX
            num_table_minmax = TRADEOFF_TABLES_MINMAX
            num_hfuncs = np.linspace(
                num_hfunc_minmax[0], num_hfunc_minmax[1], num=num_experiments
            )
            num_tables = np.linspace(
                num_table_minmax[0], num_table_minmax[1], num=num_experiments
            )

            for idx in range(num_experiments):
                num_hfunc = int(num_hfuncs[idx])
                num_table = int(num_tables[idx])
                log_info = (
                    lsh_type
                    + "_"
                    + "hfunc"
                    + str(num_hfunc)
                    + "_num_table"
                    + str(num_table)
                )
                start_time = time.time()
                candidates_array, budget_array = run_lsh_mips(
                    atoms=atoms,
                    signals=signals,
                    num_best_atoms=num_best_atoms,
                    num_hfunc=num_hfunc,
                    num_tables=num_table,
                    type=lsh_type,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array, num_signals
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is H2ALSH:
            delta_minmax = TRADEOFF_H2ALSH["powers_delta"]
            c0_minmax = TRADEOFF_H2ALSH["c0"]
            c_minmax = TRADEOFF_H2ALSH["c"]
            N0_minmax = TRADEOFF_H2ALSH["N0"]
            powers_delta = np.linspace(
                start=delta_minmax[0],
                stop=delta_minmax[1],
                num=num_experiments,
                endpoint=True,
            )
            deltas = 1 / 2 * 2 ** powers_delta
            c0_list = np.geomspace(
                start=c0_minmax[0],
                stop=c0_minmax[1],
                num=num_experiments,
                endpoint=True,
            )
            c_list = np.geomspace(
                start=c_minmax[0], stop=c_minmax[1], num=num_experiments, endpoint=True
            )
            N0s = np.geomspace(
                start=N0_minmax[0],
                stop=N0_minmax[1],
                num=num_experiments,
                endpoint=True,
            )
            for idx in range(num_experiments):
                delta = deltas[idx]
                c0 = c0_list[idx]
                c = c_list[idx]
                N0 = int(N0s[idx])
                start_time = time.time()
                h2alsh = H2ALSH_MIPS(atoms=atoms, delta=delta, c=c, c0=c0, N0=N0)
                candidates_array, budget_array = h2alsh.mip_search_queries(
                    queries=signals, top_k=num_best_atoms,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array, num_signals
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
            log_info = "_epsilon" + str(epsilon) + "_delta" + str(delta)
        elif mips_alg is NEQ_MIPS:
            num_codewords = np.geomspace(
                TRADEOFF_CODEWORD_MINMAX[0],
                TRADEOFF_CODEWORD_MINMAX[1],
                num_experiments,
            ).astype(np.int64)
            num_codebooks = np.linspace(
                TRADEOFF_CODEBOOK_MINMAX[0],
                TRADEOFF_CODEBOOK_MINMAX[1],
                num_experiments,
            ).astype(np.int64)

            for idx in range(num_experiments):
                num_codebook = num_codebooks[idx]
                num_codeword = num_codewords[idx]
                log_info = (
                    "_codebook" + str(num_codebook) + "_codeword" + str(num_codeword)
                )
                quantizer = NormPQ(n_percentile=num_codeword, quantize=PQ(M=num_codebook, Ks=num_codeword))

                start_time = time.time()
                candidates_array, budget_array = execute(
                    top_k=num_best_atoms,
                    pq=quantizer,
                    X=atoms,
                    Q=signals.astype("float32"),
                    G=naive_candidates_array,
                    num_codebooks=num_codebook,
                    num_codewords=num_codeword,
                    train_size=num_atoms,
                    seed=seed,
                )
                # includes the time finding recall (other baselines don't)
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(
                    naive_candidates_array, candidates_array
                )
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is NAPG_MIPS:
            Ms = np.geomspace(4, 32, num_experiments).astype(np.int64)
            ef_constructions = np.geomspace(2, num_atoms // 2, num_experiments).astype(np.int64)
            ef_searches = np.geomspace(2, num_atoms // 2, num_experiments).astype(np.int64)

            for (M, ef_construction, ef_search) in zip(Ms, ef_constructions, ef_searches):
                log_info = (
                        "ef_construction"
                        + str(ef_construction)
                )

                start_time = time.time()
                candidates_array, budget_array = hnsw(
                    atoms=atoms,
                    signals=signals,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                    num_links_per_node=M,
                    num_best_atoms=num_best_atoms,
                    use_norm_adjusted_factors=True
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array, num_signals)
                accuracies.append(np.sum(accuracy_array) / num_signals)
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        else:
            raise NotImplementedError(f"{mips_alg} is not implemented")

        # Add all the quantities to take their means later by dividing by the number of seeds. This is valid for all
        # quantities except accuracy_std. Going to leave for now since we don't plot errors bars in tradeoff experiments
        # TODO(@motiwari): Fix this. See #83
        avg_budget += np.array(budgets)
        avg_accuracy += np.array(accuracies)
        avg_accuracy_std += np.array(accuracies_std)
        avg_runtime += np.array(runtimes)

    avg_budget /= num_seeds
    avg_accuracy /= num_seeds
    avg_accuracy_std /= num_seeds
    avg_runtime /= num_seeds

    if add_noise:
        noise_factor = NOISE_VAR
    else:
        noise_factor = 0.0

    if is_log:
        log_dict = {
            "data_type": data_type,
            "budgets": avg_budget,
            "accuracy": avg_accuracy,
            "accuracy_std": avg_accuracy_std,
            "runtime": avg_runtime,
            "naive_budget": num_atoms * len_signals,
        }
        log_df = pd.DataFrame(log_dict)
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "tradeoff", "logs") if dir_name is None else dir_name
        filename = os.path.join(
            log_dir,
            mips_alg + "_"
            "SPEEDUP_"
            + data_type
            + "_"
            + "atoms"
            + str(num_atoms)
            + "signal_size"
            + str(len_signals)
            + "_"
            + "topk"
            + str(num_best_atoms)
            + "_"
            + log_info
            + "_noise_factor"
            + str(noise_factor)
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)

    print(f"=> Finished {mips_alg} for {data_type} \n")
