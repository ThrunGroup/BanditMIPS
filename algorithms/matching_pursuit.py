import numpy as np
import numba as nb
from math import sqrt
from scipy import linalg
from typing import Tuple
from collections import defaultdict
from scipy.linalg.lapack import get_lapack_funcs
from sklearn.linear_model import OrthogonalMatchingPursuit  # for debugging purposes

from algorithms.mips_bandit import mips_bandit
from data.get_data import get_data
from utils.utils import set_seed, get_recon_error
from utils.constants import (
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    MOVIE_LENS,
    NETFLIX,
    ACTION_ELIMINATION,
    DEFAULT_MAXMIN,
    RECON_ERROR_THRESHOLD,
    SIMPLE_SONG,
)


def matching_pursuit(
    atoms: np.ndarray,
    signal: np.ndarray,
    n_nonzero_coefs: int = 5,
    min_residual: float = 0.001,
    bandit_alg: str = "",
    seed: int = 0,
    abs: bool = False,
):
    """
    Runs the Matching Pursuit algorithm. Refer to implementation by https://github.com/stonemason11
    :param atoms: Atoms array
    :param signal: Signal vector
    :param n_nonzero_coefs: number of decompositions we want to make (equivalent to scikit-learn's n_nonzero_coefs)
    :param min_residual: the threshold for which to terminate
    :param bandit_alg: Name of the bandit algorithm to get MIPS. Default naive.
    :param seed: Seed
    :param abs: Whether to take absolute value when computing maximal inner product search.
    :return: an array of indices similar to scikit-learn OMP's indices attribute AND budget
    """
    set_seed(seed)
    decomp = 0
    total_budget = 0
    unexplored = np.arange(atoms.shape[0])
    candidates = np.zeros(n_nonzero_coefs, dtype=np.int64)
    residual = signal

    while decomp < n_nonzero_coefs and la.norm(residual) > min_residual:
        # solve the MIPS problem
        if len(bandit_alg) > 0:
            candidate, budget = mips_bandit(
                atoms[unexplored],  # exclude previous decompositions
                np.expand_dims(residual, axis=0),  # need to pass in 2d array
                maxmin=(np.min(atoms), np.max(atoms)),
                bandit_alg=bandit_alg,
                epsilon=0.01,
                abs=abs,
                delta=0.01,
            )
        else:
            if abs:
                candidate = np.argmax(np.abs(atoms[unexplored] @ residual))
            else:
                candidate = np.argmax(atoms[unexplored] @ residual)
            budget = atoms[unexplored].shape[0] * atoms[unexplored].shape[1]

        # update candidates and exclude them from atoms
        candidates[decomp] = candidate
        unexplored = np.delete(unexplored, candidate, axis=0)
        total_budget += budget
        decomp += 1

        # update the residual
        residual -= (residual @ atoms[candidate]) * atoms[candidate]

    return candidates, total_budget


# Todo: currently only supports single signal...
def orthogonal_matching_pursuit(
    atoms: np.ndarray,
    signal: np.ndarray,
    n_nonzero_coefs: int = 5,
    min_residual: float = 0.001,
    bandit_alg: str = "",
    seed: int = 0,
    abs: bool = False,
    batch_size: int = 30,
    maxmin: Tuple[float, float] = (10.0, 0.0),

    # caching params
    use_cache: bool = True,
    use_naive: bool = False,
    cache_multiplier: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Runs the Orthogonal Matching Pursuit algorithm.
    Most of the codes were borrowed from scikit-learn
    See https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_omp.py#L29

    :param atoms: Atoms array
    :param signal: Signal vector
    :param n_nonzero_coefs: number of decompositions we want to make (equivalent to scikit-learn's n_nonzero_coefs)
    :param min_residual: the threshold for which to terminate
    :param bandit_alg: Name of the bandit algorithm to get MIPS. Default naive.
    :param seed: Seed
    :param abs: Whether to take absolute value when computing maximal inner product search.
    :param batch_size: the batch_size that will be passed down to the MIPS algorithm
    :param maxmin: the del
    :return: an array of indices similar to scikit-learn OMP's indices attribute AND budget
    """
    if use_naive:
        assert use_cache == True, "cannot use naive-cache if use_cache isn't specified"

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    atoms = atoms.copy("F")
    num_atoms, num_dimensions = atoms.T.shape

    total_budget = 0  # BANDIT

    min_float = np.finfo(atoms.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (atoms,))
    (potrs,) = get_lapack_funcs(("potrs",), (atoms,))

    alpha = np.dot(atoms.T, signal)
    residual = signal
    n_active = 0
    indices = np.arange(num_atoms)  # keeping track of swapping

    # need to set these default values even if we don't use cache due to numba type errors
    max_cache_size = int((cache_multiplier * num_dimensions))
    shuffled_indices = np.arange(num_dimensions)
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    cache = np.empty((num_atoms, max_cache_size))
    cache_tracker = np.zeros(num_atoms, dtype=np.int64)  # track how many cached values each atom has
    cache_map = nb.typed.List()
    for i in range(num_atoms):
        cache_map.append(nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64))

    L = np.empty((n_nonzero_coefs, n_nonzero_coefs), dtype=atoms.dtype)
    iter = 0
    while np.linalg.norm(residual) > min_residual:

        # solve the MIPS problem
        if len(bandit_alg) > 0:
            # Todo: only supports caching with Action Elimination for now
            iter += 1
            print(f"MP iteration {iter}\n => residual is {np.linalg.norm(residual)}\n => maxmin is {maxmin}\n")

            # naive-caching uses different permutations for each MP iteration.
            # this makes it less likely that we'll get cache hits compared to PI caching.
            if use_naive:
                np.random.shuffle(shuffled_indices)

            candidate, budget, cache, cache_tracker, cache_map = mips_bandit(
                atoms.T,
                residual.T,  # need to pass in 2d array
                maxmin=maxmin,
                bandit_alg=bandit_alg,
                epsilon=0.0,
                abs=abs,
                delta=0.01,
                use_cache=use_cache,
                shuffled_indices=shuffled_indices,
                cache=cache,
                cache_tracker=cache_tracker,
                cache_map=cache_map,
                seed=seed,
            )
            lam = candidate[0][0]
            #print("Identified frequency", lam)

            # update params
            cache = cache
            cache_tracker = cache_tracker
            cache_map = cache_map
        else:
            if abs:
                lam = np.argmax(np.abs(atoms.T @ residual))
            else:
                lam = np.argmax((atoms.T @ residual))
            budget = atoms.shape[0] * atoms.shape[1]

        total_budget += budget
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # atom already selected or inner product too small
            print("=> atom already selected")
            break

        if n_active > 0:
            # Updates the Cholesky decomposition of atoms' atoms
            L[n_active, :n_active] = np.dot(atoms[:, :n_active].T, atoms[:, lam])
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = linalg.norm(atoms[:, lam]) ** 2 - v
            if Lkk <= min_float:  # selected atoms are dependent
                print("selected atoms are dependent")
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = linalg.norm(atoms[:, lam])

        # @Todo need to account for this swapping in caching
        atoms.T[n_active], atoms.T[lam] = swap( atoms.T[n_active], atoms.T[lam])
        alpha[n_active], alpha[lam] = swap(alpha[n_active], alpha[lam])
        indices[n_active], indices[lam] = swap(indices[n_active], indices[lam])
        n_active += 1

        # solves LL'atoms = atoms'signal as a composition of two triangular systems
        gamma, _ = potrs(
            L[:n_active, :n_active], alpha[:n_active], lower=True, overwrite_b=False
        )

        # need to update residual and maxmin since delta (i.e. arm distances) will get smaller
        # TODO(@motiwari): Right now this is only a heuristic, may be wrong!
        projection = np.dot(atoms[:, :n_active], gamma)
        residual = signal - projection
        decrease_by = np.linalg.norm(residual) / np.linalg.norm(signal)
        maxmin = (maxmin[0] * decrease_by, maxmin[1] * decrease_by)
        if n_active == n_nonzero_coefs:
             break

    indices = indices[:n_active]
    return indices, gamma, total_budget


def main(seed, num_atoms, len_signal, num_signals):
    np.random.seed(seed)
    atoms, signals = get_data(
        num_atoms=num_atoms,
        len_signal=len_signal,
        seed=seed,
        data_type=SIMPLE_SONG,
        num_signals=num_signals,
    )

    # scikit-learn implementation
    model1 = OrthogonalMatchingPursuit(normalize=False, n_nonzero_coefs=5, fit_intercept=False, precompute=False)
    model1.fit(atoms.T, signals[0])
    sklearn_indices = model1.coef_.nonzero()[0]

    # our naive implementation
    naive_candidates, gamma, budget = orthogonal_matching_pursuit(
        atoms=atoms.T,
        signal=signals[0],
        n_nonzero_coefs=5,
        min_residual=0.01,
        bandit_alg="",
        abs=True,
        seed=seed,
        use_cache=False,
    )
    naive_recon = get_recon_error(gamma, atoms[naive_candidates], signals[0])

    # bandit implmentation
    bandit_candidates, gamma, budget = orthogonal_matching_pursuit(
        atoms=atoms.T,
        signal=signals[0],
        n_nonzero_coefs=5,
        min_residual=0.01,
        bandit_alg=ACTION_ELIMINATION,
        abs=True,
        seed=seed,
        use_cache=False,
        use_naive=False,
        cache_multiplier=1,
    )
    bandit_recon = get_recon_error(gamma, atoms[bandit_candidates], signals[0])

    print("RECON ERRORS")
    print("bandit recon error:", bandit_recon)
    print("naive recon error:", naive_recon)
    sklearn_recon = get_recon_error(model1.coef_[:4].reshape(-1, 1), atoms[:4], signals[0])
    print("sklearn recon error WITH ONLY 5 FOURIER COMPONENTS", sklearn_recon)

    assert len(np.intersect1d(sklearn_indices, naive_candidates)) == len(sklearn_indices), \
        "The selected indices for our naive implementation and sklearn's OMP model are not identical"
    assert len(np.intersect1d(naive_candidates, bandit_candidates)) == len(naive_candidates), \
        "The selected indices for our naive implementation and sklearn's OMP model are not identical"
    assert naive_recon < RECON_ERROR_THRESHOLD, \
        f"The reconstruction error is greater than {RECON_ERROR_THRESHOLD}"
    assert bandit_recon < RECON_ERROR_THRESHOLD, \
        f"The reconstruction error is greater than {RECON_ERROR_THRESHOLD}"
    print("passed all tests")


if __name__ == "__main__":
    seed = 0
    num_atoms = 10
    len_signal = int(26e6)
    num_signals = 1
    main(seed, num_atoms, len_signal, num_signals)
