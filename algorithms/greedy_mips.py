import os
import numba as nb
import numpy as np
import heapq
import time
from typing import Tuple

from utils.utils import fit_and_plot, r, b
from utils.constants import (
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    UNIFORM_PAPER,
    NORMAL_PAPER,
    GREEDY_MIPS,
    GREEDY_ATOM_SIZE,
    GREEDY_QUERY_SIZE,
)


def generate_conditer(atoms: np.ndarray) -> np.ndarray:
    """
    Query-independent pre-processing procedure in Greedy-MIPS

    :param atoms: Atoms matrix (2d-array)
    :return: Sorted indices of each column
    """
    return np.argsort(-atoms, axis=0)


@nb.njit
def generate_heap(atoms: np.ndarray, conditer: np.ndarray, signal: np.ndarray):
    """
    Query-dependent pre-processing procedure in Greedy-MIPS.

    Heap contains (z, t) where z is the maximum value of inner product between atoms' t-th row and signal's t-th
    element for every t in range(length of signal).
    """
    for t in range(signal.shape[0]):
        if signal[t] > 0:
            z = -atoms[conditer[0, t], t] * signal[t]  # To make max-heapq
        else:
            z = -atoms[conditer[-1, t], t] * signal[t]

        if t == 0:
            Q = [(z, t)]
        else:
            heapq.heappush(Q, (z, t))
    return Q, np.zeros(signal.shape[0], dtype=np.int32)


@nb.njit
def candidate_screening(
    atoms: np.ndarray,
    signal: np.ndarray,
    conditer: np.ndarray,
    iters: np.ndarray,
    Q: nb.typed.List,
    budget: int,
    visited_j: np.ndarray,
):
    """
    Screening candidates that have maximum one-element inner product with signal. See page 6 in
    https://proceedings.neurips.cc/paper/2017/file/39d352b0395ba768e18f042c6e2a8621-Paper.pdf.
    """
    candidates = []
    complexity = int(0)
    while len(candidates) < budget:

        # Pop the maximum element from heap and find its row index in atom (column index is given)
        # using conditer array(argsort the columns of atoms)
        z, t = heapq.heappop(Q)
        z = -z  # heapq is min heap but regard it as max heap
        conditer_idx = iters[t]
        j = (
            conditer[conditer_idx, t]
            if signal[t] > 0
            else conditer[-1 - conditer_idx, t]
        )  # Index from the last if signal is negative
        complexity += 3 + int(np.log2(len(Q)))

        # j is the index of atom that has k-th largest one-element inner product with signal (k=time step of while loop)
        if visited_j[j] == 0:
            candidates.append(j)
            visited_j[j] = 1
            complexity += 1

        # Push new atom index to heap since we pop one item previously
        while conditer_idx < conditer.shape[0]:
            iters[t] += 1
            conditer_idx = iters[t]
            j = (
                conditer[conditer_idx, t]
                if signal[t] > 0
                else conditer[-1 - conditer_idx, t]
            )
            complexity += 3

            if visited_j[j] == 0:
                z = atoms[j, t] * signal[t]
                heapq.heappush(Q, (-z, t))
                complexity += 1 + int(np.log2(len(Q)))
                break
    return np.array(candidates), complexity


def greedy_mips(
    atoms: np.ndarray,
    signals: np.ndarray,
    budget: int = 10,
    verbose: bool = False,
    num_best_atoms: int = 1,
    conditer: np.ndarray = None,
    is_check: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Greedy MIPS
    - Description: Find atoms(the number of atoms is set by budget param) that have maximum one-element product
    with signal. With these atoms, find the best atom(s) by naively computing inner product.

    :param atoms: Atoms array
    :param signals: Signals array
    :param budget: Number of atoms of which we naively compute inner product
    :param verbose: Whether to
    :param num_best_atoms: Number of the best atoms we choose
    :param conditer: Preprocessed array that is not signal dependent.
    :param is_check: Whether to check the algorithm well behaves. Turn off as it increases runtime.
    :return: Arrays of final answers of MIPS, signal(query)-dependent runtime, signal-independent runtime, and
    number of calculations.
    """
    assert (budget <= atoms.shape[0])
    start_time = time.time()

    # Query-independent pre-process
    if conditer is None:
        conditer = generate_conditer(atoms)

    if verbose:
        print(f"preprocess time: {time.time() - start_time}")

    visited_j = np.zeros(atoms.shape[0])
    best_atom_list = []
    query_dependent_time_list = []
    candidate_ranking_time_list = []
    total_complexity_list = []

    for signal in signals:
        query_dependent_start_time = time.time()

        # Query dependent preprocess
        heap, iters = generate_heap(atoms, conditer, signal)
        generate_heap_complexity = int(signal.shape[0]) * int(np.log2(signal.shape[0]))
        heap = nb.typed.List(heap)

        # Candidate Screening
        candidates, screen_complexity = candidate_screening(
            atoms, signal, conditer, iters, heap, budget, visited_j
        )
        query_dependent_time = time.time() - query_dependent_start_time

        # Rank the candidates
        ranking_start_time = time.time()
        best_atom_list.append(
            candidates[np.argsort(-atoms[candidates] @ signal)[:num_best_atoms]]
        )
        ranking_complexity = budget * (
            int(signal.shape[0]) * (int(np.log2(signal.shape[0])) + 1)
        )
        candidate_ranking_time = time.time() - ranking_start_time

        # log the results
        total_complexity_list.append(
            generate_heap_complexity + screen_complexity + ranking_complexity
        )
        query_dependent_time_list.append(query_dependent_time)
        candidate_ranking_time_list.append(candidate_ranking_time)

        for candidate in candidates:
            visited_j[candidate] = 0

        if is_check:
            assert (
                np.sum(
                    candidates
                    == np.argsort(
                        -np.max(atoms * np.expand_dims(signal, axis=0), axis=1)
                    )[:budget]
                )
                / budget
                == 1.0
            ), "Greedy MIPS algorithm is broken"

        if verbose:
            print(
                f"query dependent preprocess + candidates screening: {query_dependent_time}"
            )
            print(f"candidate ranking: {candidate_ranking_time}")
    return (
        np.array(best_atom_list),
        np.array(query_dependent_time_list),
        np.array(candidate_ranking_time_list),
        np.array(total_complexity_list),
    )


def reproduce_figure6(seed: int = 0):
    num_experiments = 50
    num_best_atoms = 5
    num_naive_best_atoms = 20
    rng = np.random.default_rng(seed)
    for n in GREEDY_ATOM_SIZE:
        for k in GREEDY_QUERY_SIZE:
            speedup_list = []
            accuracy_list = []
            time_list = []
            naive_best_arms = []

            # Dataset described in https://arxiv.org/pdf/1610.03317.pdf#page=16
            atoms = rng.normal(size=(2**n, 2**k))
            signals = rng.normal(size=(num_experiments, 2**k))

            # Naively compute inner product
            for signal in signals:
                start_time = time.time()
                naive_best_arms.append(
                    np.argsort(-atoms @ signal)[:num_naive_best_atoms]
                )
                time_list.append(time.time() - start_time)
            time_list = np.array(time_list)

            # Compute inner product with Greedy MIPS with various parameters
            for budget in np.geomspace(2, 500, 20):
                conditer = generate_conditer(atoms)
                (
                    best_arms,
                    query_dependent_time_list,
                    ranking_time_list,
                    _,
                ) = greedy_mips(
                    atoms,
                    signals,
                    int(2**n / budget),
                    num_best_atoms=num_best_atoms,
                    conditer=conditer,
                )

                # Compute accuracy
                accuracy = 0
                best_arms = np.array(best_arms)
                for idx in range(len(best_arms)):
                    accuracy += len(
                        np.intersect1d(naive_best_arms[idx], best_arms[idx])
                    )
                accuracy /= len(best_arms) * len(best_arms[0])

                accuracy_list.append(accuracy)
                speedup_list.append(
                    np.mean(time_list)
                    / (
                        np.mean(query_dependent_time_list[1:])
                        + np.mean(ranking_time_list[1:])
                    )
                )

            fit_and_plot(
                np.array(speedup_list),
                np.array(accuracy_list),
                None,
                None,
                ys_std=np.zeros(1),
                title=f"Greedy MIPS: $n = 2^{b(n)}, k = 2^{b(k)}$",
                xlabel=f"Speed up over naive approaches({r(np.mean(time_list))}s)",
                ylabel=f"Performance (prec @ {num_best_atoms})",
            )


if __name__ == "__main__":
    from exps.scaling_exp import scaling_exp  # To avoid circular import

    for dataset in [ADVERSARIAL_CUSTOM, NORMAL_CUSTOM, UNIFORM_PAPER, NORMAL_PAPER]:
        scaling_exp(
            epsilon=None,
            delta=None,
            data_type=dataset,
            mips_alg=GREEDY_MIPS,
            size_minmax=(10 ** 3, 10 ** 4),
            num_experiments=10,
            dirname=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), os.path.join("exps", "logs")
            ),
            num_signals=3,
            is_logspace=True,
        )
    reproduce_figure6()
