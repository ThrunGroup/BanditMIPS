import numpy as np
import numba as nb
import math
from typing import Tuple

from utils.utils import set_seed


@nb.njit
def median_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    is_experiment: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Median Elimination algorithm + Sampling without replacement, see algorithm 1 in shorturl.at/adPR3

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    len_signal = signals.shape[1]
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)

    def m(u):
        """
        A helper function for calculating the optimal number of mnist
        """
        return min(
            (u + 1) / (1 + u / len_signal),
            (u + u / len_signal) / (1 + u / len_signal),
        )
    epsilon = max(epsilon, 0.001)
    delta = max(delta, 0.001)
    set_seed(seed)
    len_signal = signals.shape[1]
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)

    for signal_idx in range(signals.shape[0]):
        signal = signals[signal_idx]
        candidates = np.arange(atoms.shape[0])
        estimates = np.zeros(atoms.shape[0])
        population_idcs = np.arange(len_signal)
        new_epsilon = epsilon / 4
        new_delta = delta / 2
        prev_T = 0  # A helper variable for calculating the optimal number of mnist in each round
        total_num_samples = 0
        budget = 0  # Total number of calculations

        while len(candidates) > num_best_atoms:
            # Calculate the optimal number of mnist in the current round
            new_T = m(
                2
                * math.log(
                    2
                    * (len(candidates) - num_best_atoms)
                    / (
                        new_delta
                        * (math.floor(((len(candidates)) - num_best_atoms) / 2) + 1)
                    )
                )
                / (new_epsilon**2)
            )
            current_num_samples = int(new_T - prev_T)
            prev_T = new_T
            budget += current_num_samples * len(candidates)

            # Sample without replacement using Fisher-Yates shuffle.
            # Time complexity is linear in len(current_num_samples)
            if not is_experiment:
                idcs = np.random.choice(
                    len_signal - total_num_samples,
                    current_num_samples,
                    replace=False,
                )
            else:
                # With an adversarial dataset, we do a worst-case experiment by not shuffling a data
                # Ex) atoms = np.transpose(np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                #                                         [1, 1, 1, 1, 0, 0, 0, 0]])
                #     signal = np.ones(8)
                #
                #     In this case, sampling without shuffling will cause a poor performance since
                #     a first few elements of each atom doesn't represent their distribution well.
                #     So, it's a worst case scenario.
                idcs = np.arange(current_num_samples, dtype=np.int64)
            sample_idcs = population_idcs[idcs]
            population_idcs = np.delete(population_idcs, idcs)

            estimates[candidates] = (
                estimates[candidates] * total_num_samples
                + (atoms[:, sample_idcs][candidates] @ signal[sample_idcs])
            ) / (total_num_samples + current_num_samples)

            candidates = candidates[
                np.argsort(estimates[candidates])
            ]  # sort candidates with estimates
            candidates = candidates[math.ceil((len(candidates) - num_best_atoms) / 2) :]
            new_epsilon *= 3 / 4
            new_delta *= 1 / 2
            total_num_samples += current_num_samples
        if verbose:
            print(f"budget is {budget}")
        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budget
    return candidates_array, budgets_array
