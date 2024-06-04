import numpy as np
from typing import Tuple

from utils.utils import set_seed
from utils.constants import (
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    BUCKET_ACTION_ELIMINATION,
    HOEFFDING,
    BATCH_SIZE,
)
from algorithms.median_elimination import median_elimination
from algorithms.action_elimination import action_elimination
from algorithms.adaptive_action_elimination import adaptive_action_elimination
from algorithms.bucket_ae import bucket_action_elimination


def mips_bandit(
    atoms: np.ndarray,
    signals: np.ndarray,
    maxmin: np.ndarray,
    bandit_alg: str,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    seed: int = 0,
    is_experiment: bool = False,
    abs: bool = False,
    verbose: bool = False,
    with_replacement: bool = False,
    var_proxy_override: bool = False,
    batch_size: int = BATCH_SIZE,
    use_cache: bool = False,
    shuffled_indices: list = None,
    cache: np.ndarray = None,
    cache_tracker: np.ndarray = None,
    cache_map: list = None,
    bucket_num_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve MIPS(Maximum Inner Product Search) problem using a multi-armed bandit algorithm with fixed confidence setting.
    MIPS problem: To find a best atom vector in data_atoms set that has maximum inner product with a signal vector.

    :param atoms: 2d array that contains atoms in rows. Atoms are the vectors one of which will be selected
                    as the vector with maximum inner product with a signal vector.
    :param signals: A matrix whose each row contains a signal vector.
    :param maxmin: A tuple with two elements which are the maximum and minimum of data_atoms array
    :param bandit_alg: A name of multi-armed bandit algorithm
    :param epsilon: A maximum additive error between the true best arm and the best arm identified by our
                    bandit algorithm
    :param delta: A probability that the difference between the reward of the true best arm and the identified
                  best arm is beyond epsilon
    :param num_best_atoms: Number of atoms that bandit algorithm identify as top/best arms.
    :param seed: A random seed
    :param is_experiment: Whether to do worst-case experiment
    :param verbose: Verbosity of function
    :param with_replacement: Whether to sample with replacement
    :param var_proxy_override: Whether to override the given var proxy (Used for adjusting the var proxy according to
                                the median value of a signal for a crypto pairs dataset)
    :param batch_size: Number of coordinates of an atom to sample at each iteration
    :param bucket_num_samples: number of elements in each bucket for bucket action elimination algorithm
    :return: Indices of best atom that have maximum inner product with a signal vector and the budget (total number of
            calculations)
    """
    assert len(atoms.shape) == 2, "Invalid size of data atoms matrix"
    assert len(signals.shape) == 2, "Invalid size of signal vector"
    assert atoms.shape[1] == signals.shape[1], "Dimension of atoms and signals must be equal"
    assert with_replacement is False, "It is always advantageous to sample without replacement"

    set_seed(seed)
    if with_replacement:
        if bandit_alg is ACTION_ELIMINATION:
            return action_elimination(
                atoms=atoms,
                signals=signals,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                epsilon=epsilon,
                delta=delta,
                abs=abs,
                num_best_atoms=num_best_atoms,
                batch_size=batch_size,
                seed=seed,
            )
        else:
            raise NotImplementedError(f"{bandit_alg} is not implemented.")
    else:
        if bandit_alg is MEDIAN_ELIMINATION:
            return median_elimination(
                atoms=atoms,
                signals=signals,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
                is_experiment=is_experiment,
                seed=seed,
            )
        elif bandit_alg is ACTION_ELIMINATION:
            mips_result = action_elimination(
                atoms=atoms,
                signals=signals,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                var_proxy_override=var_proxy_override,
                maxmin=maxmin,
                epsilon=epsilon,
                delta=delta,
                abs=abs,
                num_best_atoms=num_best_atoms,
                with_replacement=False,
                seed=seed,
                batch_size=batch_size,
                use_cache=use_cache,
                permutation=shuffled_indices,
                cache=cache,
                cache_tracker=cache_tracker,
                cache_map=cache_map,
                verbose=verbose,
            )
            if use_cache:
                return mips_result
            else:
                return mips_result[:2]
        elif bandit_alg is ADAPTIVE_ACTION_ELIMINATION:
            return adaptive_action_elimination(
                atoms=atoms,
                signals=signals,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                var_proxy_override=var_proxy_override,
                maxmin=maxmin,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
                batch_size=batch_size,
            )
        elif bandit_alg is BUCKET_ACTION_ELIMINATION:
            return bucket_action_elimination(
                atoms=atoms,
                signals=signals,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                num_samples_bucket=bucket_num_samples,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
                with_replacement=False,
                batch_size=batch_size,
            )
        else:
            raise NotImplementedError(f"{bandit_alg} is not implemented.")
