import numpy as np
import math
import numpy.linalg as la
from typing import Tuple

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    COR_NORMAL_CUSTOM,
    POSITIVE_COR_NORMAL_CUSTOM,
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,
    HIGHLY_SYMMETRIC,

    SCALING_TOPK,
)


def generate_custom_data(
    num_atoms: int = 10**3,
    len_signal: int = 10**4,
    num_signals: int = 1,
    num_best_atoms: int = SCALING_TOPK,
    seed: int = 0,
    data_type: str = NORMAL_CUSTOM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a custom dataset described in a bandit approach to MIPS paper.
    :param num_atoms: Number of atoms(arms)
    :param len_signal: A size of signal vector
    :param num_signals: Number of signals
    :param num_best_atoms: Number of the best atoms
    :param seed: A random seed
    :param data_type: The name of distribution from which data are drawn
    :return: A synthetic dataset
    """
    rng = np.random.default_rng(seed)
    atoms_array = np.empty((num_atoms, len_signal))

    if data_type == UNIFORM_PAPER:
        # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        signal = rng.uniform(low=0, high=2, size=(num_signals, len_signal))
        atoms_array = rng.uniform(low=0, high=2, size=(num_atoms, len_signal))
    elif data_type == NORMAL_PAPER:
        # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        signal = rng.normal(loc=2, size=(num_signals, len_signal))
        atoms_array = rng.normal(size=(num_atoms, len_signal))
    elif data_type is ADVERSARIAL_CUSTOM:
        atoms_array = np.zeros((num_atoms, len_signal))
        p = rng.random(num_atoms)
        for idx in range(num_atoms):
            atoms_array[idx][: math.floor(p[idx] * len_signal)] = 1
        signal = np.ones((num_signals, len_signal))
    elif data_type == NORMAL_CUSTOM:
        # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        means = rng.normal(size=num_atoms)
        atoms_array = np.empty((num_atoms, len_signal))
        for idx in range(num_atoms):
            atoms_array[idx] = rng.normal(loc=means[idx], size=len_signal)
        signal = rng.normal(loc=rng.normal(), size=(num_signals, len_signal))
    elif (
            data_type in [COR_NORMAL_CUSTOM, POSITIVE_COR_NORMAL_CUSTOM]
    ):  # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        # Unlike other datasets, try to create a signal that is correlated to a few atoms. Correlation coefficient
        # is drawn from a normal distribution.
        correlations = rng.normal(size=(num_signals + 1, num_atoms))
        if data_type is POSITIVE_COR_NORMAL_CUSTOM:
            correlations = np.abs(correlations)

        correlations = correlations / np.sum(np.abs(correlations), axis=0, keepdims=True)
        atoms_array = np.zeros((num_atoms, len_signal))
        signal = rng.normal(loc=rng.normal(), size=(num_signals + 1, len_signal))  # To include independent normal rv
        for idx in range(num_atoms):
            for cor_idx in range(num_signals):
                atoms_array[idx] += correlations[cor_idx, idx] * signal[cor_idx]
        atoms_array += 3 * np.expand_dims(rng.normal(size=num_atoms), axis=1)
        signal = signal[:-1, :]
    elif data_type == CLEAR_LEADER_HARD:
        signal_strength = 10
        best_atom_indices = np.random.choice(num_atoms, size=num_best_atoms, replace=False)
        signal = signal_strength * np.ones((num_signals, len_signal))
        atoms_array = np.zeros((num_atoms, len_signal))
        atoms_array[best_atom_indices] = signal_strength * np.ones(len_signal)
    elif data_type == CLEAR_LEADER_SOFT:
        signal_strength = 10
        best_atom_indices = np.random.choice(num_atoms, size=num_best_atoms, replace=False)
        signal = rng.normal(loc=signal_strength, size=(num_signals, len_signal))
        atoms_array = rng.normal(loc=0, size=(num_atoms, len_signal))
        atoms_array[best_atom_indices] = rng.normal(loc=signal_strength, size=len_signal)
    elif data_type == NO_CLEAR_LEADER:
        signal_strength = 2
        signal = rng.normal(loc=signal_strength, size=(num_signals, len_signal))
        atoms_array = rng.normal(loc=0, size=(num_atoms, len_signal))
    elif data_type == HIGHLY_SYMMETRIC:
        signal_strength = 0
        signal = rng.normal(loc=signal_strength, size=(num_signals, len_signal))
        atoms_array = rng.normal(loc=0, size=(num_atoms, len_signal))
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")

    return atoms_array, signal


def create_toy_discrete(num_atoms, mul):
    """
    This function creates a dictionary of atoms where the first three atoms are unit atoms a, b, c, and the rest of the
    atoms are zero vectors. The query vector is q = 3a + 2b + c with the objective of MP running three iterations
    to find the indices 0, 1, 2.
    """
    assert mul % 2 == 0 and mul >= 2, "mul must be a multiple of 2 and greater or equal to 2"
    a = np.array([1, 1, 1] * mul, dtype='float64')
    b = np.array([-1, 0, 1] * mul, dtype='float64')
    c = np.concatenate((np.array([1, 0, 1] * (mul//2)), np.array([-1, 0, -1] * (mul//2))), dtype='float64')
    random_arr = np.random.uniform(low=-1, high=1, size=(num_atoms-3, len(a)))
    atoms = np.vstack(
        (a/la.norm(a), b/la.norm(b), c/la.norm(c), random_arr/la.norm(random_arr, axis=1)[:, np.newaxis])
    )
    query = 3*a + 2*b + c
    return atoms, query


def create_toy_waves(num_atoms, d):
    """
    This function creates a dictionary of atoms of sinusoid waves where the first three sine waves a, b, c are orthogonal
    to each other with the same amplitude and the rest of the atoms are zero vectors.
    The query vector is q = 3a + 2b + c.
    """
    amplitude = 2
    t = np.linspace(0, 2 * np.pi * d, d)
    a = amplitude * np.sin(t)
    b = amplitude * np.sin(2 * t)
    c = amplitude * np.sin(3 * t)

    random_arr = np.random.uniform(low=-amplitude, high=amplitude, size=(num_atoms-3, d))
    atoms = np.vstack(
        (a/la.norm(a), b/la.norm(b), c/la.norm(c), random_arr/la.norm(random_arr, axis=1)[:, np.newaxis])
    )
    query = 3*a + 2*b + c
    return atoms, query