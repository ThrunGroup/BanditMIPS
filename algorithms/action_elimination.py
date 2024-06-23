import numpy as np
import numba as nb
from typing import Tuple
from numba import typed


from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    ACTION_ELIMINATION,
    HOEFFDING,
    SCALING_NUM_ATOMS,
    TEMP_SIZE,
)
from utils.utils import subset_2d, subset_2d_cached, fit_and_plot, set_seed
from utils.bandit_utils import get_ci


@nb.njit
def action_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    maxmin: np.ndarray,
    epsilon: float,
    delta: float,
    var_proxy_override: bool = False,
    num_best_atoms: int = 2,
    abs: bool = False,
    verbose: bool = False,
    with_replacement: bool = False,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,

    # caching params
    use_cache: bool = False,
    permutation: np.ndarray = [],
    cache: np.ndarray = [],
    cache_tracker: np.ndarray = [],
    cache_map: nb.typed.List = ({}),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with the highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2. You can set the flag
    to use_cache for more efficiency. The params for caching are as follows:

    :params permutation: the sequence of indices used for PI caching (default None i.e. naive caching)
    :params cache: the cache of size (N, max_cache_size)
    :params cache_tracker: array of size N that tracks each atom and how much is cached (< max_cache_size).
    :params maxmin : Array of max and min values that the data can have

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)

    # run AE for all signals (this is the to find the confidence intervals over multiple experiments)
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

        # reset all params
        candidates = np.arange(D)  # arms (atoms)
        lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
        means = np.zeros(D)  # estimated means of arm returns
        hits = 0

        # Need to initialize cache params otherwise numba throws error...
        if not use_cache:
            permutation = np.arange(d)
            np.random.shuffle(permutation)
            cache = np.empty((D, d))
            cache_tracker = np.zeros(D, dtype=np.int64)
            cache_map = typed.List([typed.Dict.empty(key_type=nb.int64, value_type=nb.int64) for i in range(D)])

        while len(candidates) > num_best_atoms:
            t += 1
            start = (t - 1) * batch_size
            end = t * batch_size

            # Compute the true inner product of the candidates if we have more samples than dimension d.
            # (i.e. no need to approximate from this point forward)
            if end > d:
                sample_indices = permutation[start:]
                signal_sampled = signal[sample_indices]
                candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_indices)
                means[candidates] = (
                    (end * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - start) * len(candidates)

                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]

                # At this point, we've computed all possible inner products.
                # Exit the loop.
                break

            sample_indices = permutation[start: end]
            candidates_sampled, budget, _cache, _cache_tracker, _cache_map, _hits = subset_2d_cached(
                source_array=atoms,  # this changes per OMP iteration
                rows=candidates,
                columns=sample_indices,
                use_cache=use_cache,
                cache=cache,
                cache_tracker=cache_tracker,
                cache_map=cache_map,
            )
            # update params
            budgets += budget
            cache = _cache
            cache_tracker = _cache_tracker
            cache_map = _cache_map
            hits += _hits

            signal_sampled = signal[sample_indices]
            means[candidates] = (
                (end * means[candidates])
                + (candidates_sampled @ signal_sampled)
            ) / (
                (t + 1) * batch_size
            )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?

            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=end,
                pop_size=d,
                with_replacement=False,
            )
            if 4 * ci < epsilon:
                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            # update the candidates' upper and lower bounds
            u_means = means[candidates]
            l_means = means[candidates]
            if abs:
                u_means = np.abs(u_means)
                l_means = np.abs(l_means)

            # TODO(@motiwari): Make sure there are no NaNs here
            ucbs[candidates] = u_means + ci
            lcbs[candidates] = l_means - ci
            lcb_criteria = lcbs[np.argsort(-lcbs)[num_best_atoms-1]]  # Have to compare ucbs with top_k atom's lcb
            new_cand_idcs = np.where((ucbs[candidates] >= lcb_criteria))[0]
            if (
                len(new_cand_idcs) < num_best_atoms
            ):  # Have to return candidates with length num_best_atoms
                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            candidates = candidates[new_cand_idcs]
        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets

        if verbose:
            print(f"Num iterations: {t}")

    return candidates_array, budgets_array, cache, cache_tracker, cache_map
