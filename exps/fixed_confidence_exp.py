import numpy as np
import math
from typing import Tuple
import matplotlib.pyplot as plt

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
    UNIFORM_PAPER,
    NORMAL_PAPER,
)
from data.custom_data import generate_custom_data
from data.get_data import get_data

from algorithms.mips_bandit import mips_bandit


def fixed_confidence_exp(
    epsilons: Tuple[float, ...],
    deltas: Tuple[float, ...],
    num_atoms: int = 10 ** 3,
    len_signal: int = 10 ** 3,
    data_type: str = ADVERSARIAL_CUSTOM,
    bandit_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    verbose: bool = False,
    with_replacement: bool = True,
    num_best_atoms: int = 1,
) -> None:
    """
    Validate that bandit algorithm for fixed confidence setting works well. In other words, observed failure
    probability for bandit algorithm with fixed confidence setting should be lower than failure probability(delta)
    passed as a parameter for a bandit algorithm.

    :param epsilons: A tuple of epsilons
    :param deltas: A tuple of deltas
    :param data_type: A name of data generating function
    :param bandit_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments for each fixed confidence setting.
    :param verbose: verbosity of function
    :param with_replacement: Whether the algorithm mnist with replacement
    """
    for epsilon in epsilons:
        observed_deltas = []
        for delta in deltas:
            failure_counts = 0
            print(f"Doing an experiment with epsilon={epsilon} and delta={delta}")
            for seed in range(num_experiments):
                rng = np.random.default_rng(seed)
                if verbose:
                    print(f"-seed: {seed}")
                atoms, signals = get_data(
                    num_atoms=num_atoms,
                    len_signal=len_signal,
                    seed=seed,
                    data_type=data_type,
                    num_signals=5,
                )
                maxmin = (1, -1)
                if verbose:
                    print("--data generated")
                true_inner_products = atoms @ signals.transpose()
                true_best_idcs = (
                    np.argsort(-true_inner_products, axis=0)[:num_best_atoms, :]
                ).transpose()
                if verbose:
                    print("--naively solved")
                estimated_best_idcs, _ = mips_bandit(
                    atoms=atoms,
                    signals=signals,
                    maxmin=maxmin,
                    bandit_alg=bandit_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_atoms=1,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                )
                if verbose:
                    print("--stochastically solved")
                for signal_idx in range(len(signals)):
                    failure_counts += np.sum(
                        true_inner_products[estimated_best_idcs[signal_idx], signal_idx]
                        + epsilon * signals.shape[1]
                        < true_inner_products[true_best_idcs[signal_idx], signal_idx]
                    )
            observed_delta = failure_counts / (num_experiments * signals.shape[0])
            observed_deltas.append(observed_delta)
            print(f"delta: {delta}, epsilon: {epsilon}")
            print(f"observed failure probability: {observed_delta}")
            print(f"upper bound of failure probability: {delta}\n")
            assert (
                observed_delta < delta
            ), f"{bandit_alg} is theoretically wrong in fixed confidence setting"

        plt.title(f"Fixed Confidence Experiment (epsilon: {epsilon})")
        plt.xlabel("Failure probability $\delta$")
        plt.ylabel("Observed failure probability")
        plt.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.plot(deltas, deltas, label="$\delta$ = observed failure probability")
        plt.plot(deltas, observed_deltas, label=f"{bandit_alg}", marker='o')
        plt.legend(loc="upper left")
        plt.show()