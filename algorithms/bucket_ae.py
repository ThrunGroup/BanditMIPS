import numpy as np
import numba as nb
import math
import cmath
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
)
from utils.utils import subset_2d, fit_and_plot, set_seed
from utils.bandit_utils import get_ci


@nb.njit
def bucket_action_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    with_replacement: bool = True,
    seed=0,
    num_samples_bucket: int = 30,
    num_samples_norm: int = 10000,
    z_score: float = 3,  # Scipy package is normally used for calculating z-score, but it is not compatible with numba
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Put atoms into k buckets with descending order of their norm. Then, solve MIPS problem sequentially with bandit
    techniques.  If the lcb of inner product between the best atom and query is greater than the upper bound of
    remaining buckets( = norm of atom * norm of query vector), returns the best atom. This is to reduce unnecessary
    computations to search maximum inner product atom in other buckets. This is possible sicne the best atom can be
    found by considering a few atoms. Hence, for a specific distribution, the linear scaling of sample complexity in
    number of atoms can be removed.

    Note: Here, we don't compare the lcb and ucb of inner product values from each partition. Instead, we compare
    the estimated inner product values. Hence, there are no theoretical guarantees as there may be errors due to
    sampling noise.

    MIPS problem:= For each signal in signals matrix, find an atom in atoms matrix that has maximum dot product with
    the signal
    :param atoms: A 2d array that store atoms in rows.
    :param signals: A 2d array that store signals in rows
    :param var_proxy: Sub-gaussian variance of the distribution of arms (regarding an element of atom * an element of
    signal as a sub-gaussian RV)
    :param epsilon: Sub-optimality constant (allowing an epsilon-suboptimal arm to be the best arm)
    :param delta: Error probability of not getting successful MIPS
    :param num_best_atoms: Number of atoms we want to search (top k atoms in terms of dot product with a signal)
    :param verbose: Verbosity of function
    :param batch_size: Batch size of bandit algorithm
    :param with_replacement: Whether to sample with replacement when sampling arms
    :param seed: A random seed number
    :param num_samples_bucket: Number of samples in each bucket
    :param num_samples_norm: Number of samples to estimate the norm of atoms
    :param z_score: Z score to calculate the upper bound and lower bound of the norm of atoms
    :return: An array of top k atoms and sample complexity
    """
    set_seed(seed)
    N = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms
    num_samples_bucket = min(N, num_samples_bucket)
    k = min(
        int(N / num_samples_bucket), int(N / num_best_atoms)
    )  # Todo: fix hard-coded
    m = int(N / k)
    num_samples_norm = min(
        num_samples_norm, d
    )  # Number of samples from each atom to estimate its norm

    sample_idcs = np.random.randint(
        low=0,
        high=d,
        size=num_samples_norm,  # dtype=np.int64
    )
    sampled_atoms_squared = atoms[:, sample_idcs] * atoms[:, sample_idcs]
    sampled_signals_squared = signals[:, sample_idcs] * signals[:, sample_idcs]

    # TODO(vxbrandon): The lines below are not theoretically valid. Have to find a formula for finding the norm of
    #  random vectors from unknown distribution. See the above commented lines for the general idea.
    # Partition the atoms
    atoms_ub = np.sqrt(np.sum(sampled_atoms_squared, axis=1) * d / num_samples_norm)
    signals_ub = np.sqrt(np.sum(sampled_signals_squared, axis=1) * d / num_samples_norm)
    sort_idcs = np.argsort(atoms_ub)[
        ::-1
    ]  # Sorting indices of atoms in the descending order of their norms
    ub_buckets = np.empty(k)  # Upper bound of each bucket
    bucket_idcs = []
    for idx in range(k):
        ub_buckets[idx] = atoms_ub[sort_idcs[idx * m]]
        bucket_idcs.append(sort_idcs[idx * m : min((idx + 1) * m, N)])

    candidates_array = np.empty(
        (signals.shape[0], num_best_atoms), dtype=np.int64
    )  # ith row contains top k(num_best_atoms) atoms of ith signal.
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)
    for signal_idx in range(signals.shape[0]):
        signal = signals[signal_idx]
        temp_ub_buckets = ub_buckets * signals_ub[signal_idx] / d
        means = np.empty(0)
        lcbs = np.empty(0)
        candidates = np.empty(0)
        budgets = 0
        for idx in range(k):
            best_k_lcbs = lcbs[num_best_atoms - 1] if idx != 0 else -cmath.inf
            (
                bucket_candidates,
                bucket_means,
                bucket_lcbs,
                bucket_budgets,
            ) = bucket_action_elimination_helper(
                atoms[bucket_idcs[idx]],
                signal,
                var_proxy,
                epsilon,
                delta,
                num_best_atoms,
                verbose,
                batch_size,
                with_replacement,
                seed,
                best_k_lcbs,
            )
            concatenated_means = np.concatenate((means, bucket_means))
            best_k_idcs = np.argsort(concatenated_means)[::-1][:num_best_atoms]
            means = concatenated_means[best_k_idcs]
            candidates = np.concatenate(
                (candidates, bucket_idcs[idx][bucket_candidates])
            )[best_k_idcs]
            lcbs = np.concatenate((lcbs, bucket_lcbs))[best_k_idcs]
            budgets += bucket_budgets
            # if idx == k - 1:
            #     print(f"num_atoms: {len(atoms)}/ idx: {idx}/ budget: {budgets}")
            #     break
            #
            # if np.max(lcbs) > temp_ub_buckets[idx + 1]:
            #     print(f"num_atoms: {len(atoms)}/ idx: {idx}/ budget: {budgets}")
            #     break

        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets

    return candidates_array, budgets_array


@nb.njit
def bucket_action_elimination_helper(
    atoms: np.ndarray,
    signal: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    with_replacement: bool = True,
    seed=0,
    prev_lcbs: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k (num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta / D
    budgets = 0  # Number of calculations
    candidates = np.arange(D, dtype=np.int64)  # arms (atoms)
    lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
    ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
    means = np.zeros(D)  # estimated means of arm returns
    if not with_replacement:
        shuffled_idcs = np.arange(d)
        np.random.shuffle(shuffled_idcs)
    else:
        shuffled_idcs = None

    # TODO: Add exact computation if total mnist too big
    # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
    # TODO: Don't let old candidates come back?
    # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
    while len(candidates) > num_best_atoms:
        t += 1
        # Find a true inner product value of candidates if we sample more than their dimension (length of
        # atom/signal)
        if t * batch_size > d:
            if with_replacement:
                means[candidates] = atoms[candidates] @ signal
                budgets += d * len(candidates)
            else:
                sample_idcs = shuffled_idcs[(t - 1) * batch_size :]
                signal_sampled = signal[sample_idcs]
                candidates_sampled = subset_2d(
                    atoms, rows=candidates, columns=sample_idcs
                )
                means[candidates] = (
                    (t * batch_size * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - (t - 1) * batch_size) * len(candidates)

            lcbs = ucbs = means
            candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
            break

        budgets += batch_size * len(candidates)

        if with_replacement:
            sample_idcs = np.random.choice(np.arange(d), batch_size, replace=True)
        else:
            sample_idcs = shuffled_idcs[(t - 1) * batch_size : t * batch_size]

        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
        ) / (
            (t + 1) * batch_size
        )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?
        if with_replacement:
            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=t * batch_size,
            )
        else:
            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=t * batch_size,
                pop_size=d,
                with_replacement=False,
            )

        ucbs[candidates] = means[candidates] + ci
        lcbs[candidates] = means[candidates] - ci
        new_cand_idcs = np.where(
            (ucbs[candidates] >= max(prev_lcbs, lcbs[candidates].max()))
        )[0]

        if 4 * ci < epsilon:
            sort_idcs = np.argsort(-means[candidates])
            candidates = candidates[sort_idcs][:num_best_atoms]
            break

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            sort_idcs = np.argsort(-means[candidates])
            candidates = candidates[sort_idcs][:num_best_atoms]
            break

        candidates = candidates[new_cand_idcs]
    candidates_lcbs = lcbs[candidates]
    candidates_means = means[candidates]
    return candidates, candidates_means, candidates_lcbs, budgets
