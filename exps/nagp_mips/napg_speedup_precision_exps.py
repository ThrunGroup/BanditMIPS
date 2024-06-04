import os
import numpy as np
import pandas as pd
import time
import glob
from pandas import read_csv

from utils.constants import (
    # datasets
    NORMAL_CUSTOM,
    NAPG_MIPS,

    # constants
    NUM_SEEDS,
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_DIMENSION,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_TOPK,
)
from data.get_data import get_data
from algorithms.napg import hnsw
from exps.speedup_precision_exps import get_accuracy_array


def napg_speedup_precision_exps(
        data_type: str = NORMAL_CUSTOM,
        len_signals: int = TRADEOFF_DIMENSION,
        num_atoms: int = TRADEOFF_NUM_ATOMS,
        num_experiments: int = TRADEOFF_NUM_EXPERIMENTS,
        num_signals: int = TRADEOFF_NUM_SIGNALS,
        num_best_atoms: int = TRADEOFF_TOPK,
        is_log: bool = True,
        is_plot: bool = False,
        dirname: str = None,
):
    """
    Run the precision-speed tradeoff of the bandit and baseline algorithms and store them in the appropriate
    log files. Note that what we're tuning "delta" for the bandit algorithms, "budget" for greedy-mips, "depth"
    of the pca-tree for pca-mips, and "num_hfunc and num_tables" for lsh-mips.

    :param data_type: A name of data generating function
    :param len_signals: The dimension "d" of the signals
    :param num_atoms: Number of atoms
    :param num_experiments: Number of experiments
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param dirname: Directory name to log the results
    """
    print(f"=> Tradeoff experiment for NAPG with {data_type} dataset")

    log_dir = "logs" if dirname is None else dirname
    checkpoint_path = os.path.join(log_dir, "napg_checkpoints")

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, f"*{data_type}*topk{num_best_atoms}*"))
    previous_seed = len(checkpoint_files)

    for seed in range(previous_seed, NUM_SEEDS):
        print(f"... running seed {seed}")
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=len_signals,
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )

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
            accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
            accuracies.append(np.sum(accuracy_array))
            accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        # Save the experiment results for each seed as a checkpoint
        checkpoint_log_dict = {
            "data_type": data_type,
            "budgets": np.array(budgets),
            "accuracy": np.array(accuracies),
            "accuracy_std": np.array(accuracies_std),
            "runtime": np.array(runtimes),
            "naive_budget": num_atoms * len_signals,
        }

        checkpoint_log_df = pd.DataFrame(checkpoint_log_dict)
        filename = os.path.join(
            checkpoint_path,
            NAPG_MIPS
            + "_"
            + data_type
            + "_"
            + "seed"
            + str(seed)
            + "topk"
            + str(num_best_atoms)
            + ".csv",
        )
        checkpoint_log_df.to_csv(filename, index=False)

        print(f"Checkpoint for seed {seed} saved")

        # Combine the previous checkpoint data and save it in a single CSV file. This allows the user to interrupt the
        # experiments at any time and still have access to the results for plotting.
        new_checkpoint_files = glob.glob(os.path.join(checkpoint_path, f"*{data_type}*"))
        num_checkpoints = len(new_checkpoint_files)

        avg_runtime = np.zeros(num_experiments)
        avg_budget = np.zeros(num_experiments)
        avg_accuracy_std = np.zeros(num_experiments)
        avg_accuracy = np.zeros(num_experiments)

        for checkpoint_file in new_checkpoint_files:
            data = read_csv(checkpoint_file)

            budgets = data["budgets"].tolist()
            accuracies_std = data["accuracy_std"].tolist()
            accuracy = data["accuracy"].tolist()
            runtime = data["runtime"].tolist()

            avg_budget += budgets
            avg_accuracy_std += accuracies_std
            avg_accuracy += accuracy
            avg_runtime += runtime

        avg_runtime /= num_checkpoints
        avg_budget /= num_checkpoints
        avg_accuracy_std /= num_checkpoints
        avg_accuracy /= num_checkpoints

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

            filename = os.path.join(
                log_dir,
                NAPG_MIPS
                + "_"
                + "SPEEDUP"
                + "_"
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
                + ".csv",
            )
            os.makedirs(log_dir, exist_ok=True)
            log_df.to_csv(filename, index=False)

    print(f"=> Finished NAPG for {data_type} \n")
