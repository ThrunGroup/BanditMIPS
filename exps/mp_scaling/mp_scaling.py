import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from algorithms.matching_pursuit import orthogonal_matching_pursuit
from data.get_data import get_data
from utils.utils import get_sizes_arr, get_recon_error
from utils.constants import (
    NORMAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,
    TOY_SINE,
    SIMPLE_SONG,

    ACTION_ELIMINATION,
    NUM_SEEDS,
    SCALING_NUM_EXPERIMENTS,
    DIMENSION_OF_ATOMS,
    NORMALIZED_MAXMIN,
    SCALING_DELTA,
    SCALING_EPSILON,
    SCALING_SIZE_MINMAX,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
)


def scaling_exp(
    epsilon: Optional[float] = SCALING_EPSILON,
    delta: Optional[float] = SCALING_DELTA,
    data_type: str = NORMAL_CUSTOM,
    mips_alg: str = ACTION_ELIMINATION,
    num_experiments: int = SCALING_NUM_EXPERIMENTS,
    size_minmax: Tuple[int, int] = SCALING_SIZE_MINMAX,
    maxmin: Tuple[float, float] = (2.0, 0.0),
    num_atoms: int = SCALING_NUM_ATOMS,
    num_signals: int = SCALING_NUM_SIGNALS,
    num_best_atoms: int = 1,
    with_replacement: bool = True,
    is_normalize: bool = False,
    is_log: bool = True,
    is_plot: bool = True,
    is_logspace: bool = True,
    dirname: str = None,
    use_cache: bool = False,
    use_naive: bool = False,
    num_seeds: int = NUM_SEEDS,
) -> None:
    """
    Run the time complexity of the mp algorithm as the signal size scales and store them in
    the appropriate log files.

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param num_experiments: Number of experiments
    :param size_minmax: Minimum and Maximum of signal vector size
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dirname: Directory name to log the results
    """
    if use_naive:
        assert use_cache == True, "cannot use naive-cache if use_cache isn't specified"

    global budget_array

    # average over NUM_SEEDS results
    avg_runtime_ovr_seeds = np.zeros(num_experiments)
    avg_budget_ovr_seeds = np.zeros(num_experiments)
    avg_budget_std_ovr_seeds = np.zeros(num_experiments)

    avg_accuracies_ovr_seeds = np.zeros(num_experiments)    # this is in regards to the atom indices
    avg_recon_error_ovr_seeds = np.zeros(num_experiments)   # defined as (bandit_recon - naive_recon) / naive_recon
    print(f"=> Scaling experiment for MP with {mips_alg} on {data_type} dataset")

    for seed in range(num_seeds):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)
        sizes_signal = get_sizes_arr(is_logspace, size_minmax, num_experiments)
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=sizes_signal[-1],    # start with the largest signal size and index
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )

        # params that we need to update
        avg_runtime = []
        avg_budget = []
        avg_budget_std = []
        avg_accuracies = []
        avg_recon_error = []

        # currently MP only supports single signals at a time
        for size in sizes_signal:
            subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
            atoms_subset, signals_subset = (
                atoms[:, subset_indices],
                signals[:, subset_indices],
            )

            if is_normalize:
                atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
                signal_norm = np.linalg.norm(signals_subset, axis=1, keepdims=True)
                atoms_subset /= atoms_norm
                signals_subset /= signal_norm
                epsilon /= atoms_norm.max() * signal_norm
                maxmin = NORMALIZED_MAXMIN  # Assume that the information on element-wise product is not given.

            budgets = []
            budgets_std = []
            accuracies = []
            recon_error = []
            runtimes = []
            for signal in signals_subset:   # this is one signal out of num_signals signals of the same size
                # naive version
                naive_candidates, _, _ = orthogonal_matching_pursuit(
                    atoms=atoms_subset.T,
                    signal=signal,
                    bandit_alg="",
                    abs=True,
                    use_cache=False,
                    maxmin=maxmin
                )

                start_time = time.time()
                candidates, gamma, total_budget = orthogonal_matching_pursuit(
                    atoms=atoms_subset.T,
                    signal=signal,
                    bandit_alg=mips_alg,
                    abs=True,
                    use_cache=use_cache,
                    use_naive=use_naive,
                    maxmin=maxmin,
                )
                bandit_recon = get_recon_error(gamma, atoms_subset[candidates], signal)
                runtime = time.time() - start_time

                # update param lists
                budgets.append(total_budget)
                runtimes.append(runtime)
                accuracies.append(
                    len(np.intersect1d(naive_candidates, candidates)) / len(candidates)
                )
                recon_error.append(bandit_recon)
                log_info = ""
                if use_cache:
                    if use_naive:
                        log_info += "naive_cache_"
                    else:
                        log_info += "PI_cache_"

            # get avg (and std for budget) for the num_signals signals
            avg_runtime.append(np.array(np.mean(np.array(runtimes))))
            avg_budget.append(np.array(np.mean(np.array(budgets))))
            avg_budget_std.append(
                np.std(np.sum(np.array(budgets))) / np.sqrt(num_signals)
            )
            avg_accuracies.append(np.array(np.mean(np.array(accuracies))))
            avg_recon_error.append(np.array(np.mean(np.array(recon_error))))

        # keep track of params for all seeds (will find the average at the end)
        avg_runtime_ovr_seeds += avg_runtime
        avg_budget_ovr_seeds += avg_budget
        avg_budget_std_ovr_seeds += avg_budget_std
        avg_accuracies_ovr_seeds += avg_accuracies
        avg_recon_error_ovr_seeds += avg_recon_error

    avg_runtime_ovr_seeds /= num_seeds
    avg_budget_ovr_seeds /= num_seeds
    avg_budget_std_ovr_seeds /= num_seeds
    avg_accuracies_ovr_seeds /= num_seeds
    avg_recon_error_ovr_seeds /= num_seeds

    # save to log files
    if is_log:
        log_dict = {
            "signal_sizes": sizes_signal,
            "budgets": avg_budget_ovr_seeds,
            "budgets_std": avg_budget_std_ovr_seeds,
            "runtime": avg_runtime_ovr_seeds,
            "idx_accuracy": avg_accuracies_ovr_seeds,
            "rec_error": avg_recon_error_ovr_seeds,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = "logs" if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            "SCALING_"
            + data_type
            + "_"
            + "atoms"
            + str(num_atoms)
            + "_"
            + "epsilon"
            + str(epsilon)
            + "_"
            + "delta"
            + str(delta)
            + log_info
            + "_"
            + "ind_var"
            + DIMENSION_OF_ATOMS
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)
    print(f"Performance of {mips_alg} on {data_type} dataset:")
    print(f"==> accuracy of indices selected: {np.mean(avg_accuracies_ovr_seeds)}")
    print(f"==> reconstruction error: {np.mean(avg_recon_error_ovr_seeds)}")


if __name__ == "__main__":
    scaling_exp(
        data_type=SIMPLE_SONG,
        size_minmax=(2*60*44100, 10*60*44100),
        num_atoms=10,
        use_cache=False,
        use_naive=False,
        maxmin=(10.0, 0),
    )
