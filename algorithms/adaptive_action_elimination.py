import numpy as np
import numba as nb
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    HOEFFDING,
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci


@nb.njit
def adaptive_action_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    maxmin: np.ndarray,
    epsilon: float,
    delta: float,
    var_proxy_override: bool = False,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    An adaptive AE(Action Elimination) algorithm that adopts sampling without replacement, adaptive variance proxy, and
    smart sampling on top of original AE algorithm. It applies to MIPS(Maximum Inner product search) problem adaptively.

    Adaptive changes
    -Sample without replacement: Get a tight confidence interval by sampling without replacement, see lemma 1 in
    https://arxiv.org/pdf/1309.4029.pdf
    -Adaptive variance proxy: Tracking the sampled elements of signal vector and atoms, we can make the bounds of
    arms' rewards(dot product of atom and signal whose elements are sampled) tighter.
    -Smart sampling: By sorting the signal array, we can make confidence interval tighter and can expect the faster
    best arm identification.

    Should assert that all vectors are normalized. Omit this assertion statement in function due to its hevay
    computation.

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)
    for signal_idx in range(signals.shape[0]):
        signal = signals[signal_idx]

        # Some signals of a crypto pairs dataset can be small (most of the values around 0.001, for example),
        # making it hard to eliminate bad atoms.
        # Adjust maxmin according to the median value of the signal (There can be a more sophisticated way)
        if var_proxy_override:
            signal_median = np.median(signal)
            var_proxy = 1 / 4 * ((maxmin[0] - maxmin[1]) * signal_median) ** 2

        D = atoms.shape[0]  # Number of atoms
        d = atoms.shape[1]  # Dimension of atoms/signal
        t = 0  # time step
        new_delta = delta / D
        budgets = 0  # Number of calculations
        candidates = np.arange(D)  # arms (atoms)
        lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
        means = np.zeros(D)  # estimated means of arm returns
        sorted_idcs = np.argsort(
            -np.abs(signal)
        )  # Indices that sorts signal in the descending order
        sum_var_proxy = 0

        while len(candidates) > num_best_atoms:
            t += 1

            # Find a true inner product value of candidates if we sample as much as their length (length of atom/signal)
            if t * batch_size > d:
                sample_idcs = sorted_idcs[(t - 1) * batch_size :]
                signal_sampled = signal[sample_idcs]
                candidates_sampled = subset_2d(
                    atoms, rows=candidates, columns=sample_idcs
                )
                means[candidates] = (
                    (t * batch_size * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - (t - 1) * batch_size) * len(candidates)
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            batch_size = min(
                batch_size, d - (t - 1) * batch_size
            )  # Can't sample more than population size
            sample_idcs = sorted_idcs[(t - 1) * batch_size : t * batch_size]
            signal_sampled = signal[sample_idcs]
            candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
            means[candidates] = (
                (t * batch_size * means[candidates])
                + (candidates_sampled @ signal_sampled)
            ) / ((t + 1) * batch_size)

            max_signal = np.abs(signal_sampled[0])
            sum_var_proxy += var_proxy * (max_signal**2)
            ci_vec = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=(t * batch_size),
                pop_size=signal.shape[0],
                with_replacement=False,
                sum_var_proxy=sum_var_proxy,
            )
            budgets += batch_size * len(candidates)
            if 4 * ci_vec < epsilon:
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            ucbs[candidates] = means[candidates] + ci_vec
            lcbs[candidates] = means[candidates] - ci_vec
            lcb_criteria = lcbs[np.argsort(-lcbs)[num_best_atoms-1]]  # Have to compare ucbs with top_k atom's lcb
            new_cand_idcs = np.where((ucbs[candidates] >= lcb_criteria))[0]

            if (
                len(new_cand_idcs) < num_best_atoms
            ):  # Have to return candidates with length num_best_atoms
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            candidates = candidates[new_cand_idcs]
        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets
    return candidates_array, budgets_array
