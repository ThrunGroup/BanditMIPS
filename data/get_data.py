from typing import Tuple
import numpy as np
import math
import os
import torch

import data.mnist
from utils.utils import subset_2d, add_eps_noise
from data.custom_data import generate_custom_data, create_toy_waves
from data.simple_song import create_song, create_note_waves, create_sin_wave
from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,
    COR_NORMAL_CUSTOM,
    POSITIVE_COR_NORMAL_CUSTOM,
    NETFLIX_TRANSPOSE,
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,
    HIGHLY_SYMMETRIC,
    GPT2_LM_HEAD,
    OPT_LM_HEAD,
    TOY_SINE,
    TOY_DISCRETE,
    MNIST_T,
    SIMPLE_SONG,
    SIFT_1M,
    SAMPLE_RATE,
    SECONDS_PER_MINUTE,
    SCALING_TOPK,
)


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def choose_signal_and_atoms(data: np.ndarray):
    N = len(data)
    signal_idx = np.random.choice(N)
    signals = data[signal_idx].reshape(1, -1)

    if signal_idx == 0:
        atoms = data[signal_idx + 1 :]
    elif signal_idx == N - 1:
        atoms = data[:signal_idx]
    else:
        atoms = np.vstack([data[:signal_idx], data[signal_idx + 1 :]])  # TODO: Boundary

    return signals, atoms


def get_data(
    num_atoms: int = 10 ** 3,
    len_signal: int = 10 ** 4,
    num_signals: int = 1,
    num_best_atoms: int = SCALING_TOPK,
    seed: int = 0,
    data_type: str = NORMAL_CUSTOM,
    add_noise: bool = False,
    is_tradeoff: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if data_type in [
        ADVERSARIAL_CUSTOM,
        NORMAL_CUSTOM,
        NORMAL_PAPER,
        UNIFORM_PAPER,
        COR_NORMAL_CUSTOM,
        CLEAR_LEADER_HARD,
        CLEAR_LEADER_SOFT,
        NO_CLEAR_LEADER,
        POSITIVE_COR_NORMAL_CUSTOM,
        HIGHLY_SYMMETRIC,
    ]:
        return generate_custom_data(
            num_atoms=num_atoms,
            len_signal=len_signal,
            num_signals=num_signals,
            num_best_atoms=num_best_atoms,
            seed=seed,
            data_type=data_type,
        )
    elif data_type == GPT2_LM_HEAD:
        assert len_signal <= 1600, f"GPT2-XL embedding dimension is 1600 which is less than {len_signal}"
        queries_filename = os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips_gpt2"), "GPT2_QUERIES.pt"
        )
        atoms_filename = os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips_gpt2"), "GPT2_LM_HEAD_WEIGHT.pt"
        )
        queries = torch.load(queries_filename).detach().numpy().astype('float64')
        atoms = torch.load(atoms_filename).detach().numpy().astype('float64')
        sample_idcs = rng.choice(queries.shape[1], len_signal, replace=False)
        signals = queries[:num_signals, sample_idcs]
        atoms = atoms[:num_atoms, sample_idcs]
        return atoms, signals
    elif data_type == OPT_LM_HEAD:
        assert len_signal <= 4096, f"GPT2-XL embedding dimension is 1600 which is less than {len_signal}"
        queries_filename = os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips_opt"), "OPT_QUERIES.pt"
        )
        atoms_filename = os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips_opt"), "OPT_LM_HEAD_WEIGHT.pt"
        )
        queries = torch.load(queries_filename).detach().numpy().astype('float64')
        atoms = torch.load(atoms_filename).detach().numpy().astype('float64')
        sample_idcs = rng.choice(queries.shape[1], len_signal, replace=False)
        signals = queries[:num_signals, sample_idcs]
        atoms = atoms[:num_atoms, sample_idcs]
        return atoms, signals
    elif data_type == NETFLIX:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "netflix"), "Movie_ratings.npy"
        )
        data = np.load(filename)
    elif data_type == NETFLIX_TRANSPOSE:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "netflix"), "Movie_ratings.npy"
        )
        data = np.load(filename)
        data = data.transpose()
    elif data_type == MOVIE_LENS:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "movie_lens"),
            "movie_lens_1m_ratings.npy",
        )
        data = np.load(filename)
    elif data_type == CRYPTO_PAIRS:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "crypto_pairs"),
            "crypto_pairs_1m_dimensions.npy",
        )
        data = np.load(filename)
    elif data_type == TOY_SINE:
        atoms, signal = create_toy_waves(num_atoms, len_signal)
        return atoms, np.array([signal])
    elif data_type == MNIST_T:
        mndata = mnist.MNIST(os.path.join(os.path.dirname(__file__), "mnist"))
        images, labels = mndata.load_training()
        data = np.array(images)
        data = data.T
        num_atoms = 783
        num_signals = 1
    elif data_type == SIMPLE_SONG:
        assert num_atoms >= 6, "Need at least 6 atoms for the simple song's notes"
        num_mins = 10  # Creates 26,460,000-dimensional data
        song = create_song(num_minutes=num_mins)
        signals = song.reshape(1, -1)
        num_signals = 1

        C0, E0, G0, C1, E1, G1 = create_note_waves()
        G1_freq = 392
        atoms = np.vstack([C0, E0, G0, C1, E1, G1])
        for i in range(7, num_atoms):
            note_freq = math.floor(G1_freq * 2 ** ((i - 6) / 12))
            extraneous_note = create_sin_wave(
                frequency_multiplier=note_freq,
                sampling_frequency=SAMPLE_RATE,
                num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_mins,
            )
            atoms = np.vstack([atoms, extraneous_note])

        sample_idcs = rng.choice(signals.shape[1], len_signal, replace=False)
        return atoms[:, sample_idcs], signals[:, sample_idcs]
    elif data_type == SIFT_1M:
        X = fvecs_read("../../data/sift/sift_base.fvecs")
        data = X.T
        data = data.astype(np.float64)
        signals, atoms = choose_signal_and_atoms(data)
        sample_idcs = rng.choice(signals.shape[1], len_signal, replace=False)
        return atoms[:, sample_idcs], signals[:, sample_idcs]
    else:
        raise NotImplementedError(f"{data_type} is not implemented")

    assert (
        num_atoms + num_signals <= data.shape[0]
    ), "Number of atoms + number of signals given are greater than the number of rows of data"
    assert (
        len_signal <= data.shape[1]
    ), "Length of signal given is greater than the number of columns of data"

    signal_idcs = rng.choice(np.arange(data.shape[0]), num_signals, replace=False)
    sample_idcs = rng.choice(data.shape[1], len_signal, replace=False)

    atom_idcs = rng.choice(
        np.setdiff1d(np.arange(data.shape[0]), signal_idcs), num_atoms, replace=False
    )
    atoms = subset_2d(data, atom_idcs, sample_idcs)
    if (
        data_type in [NETFLIX, MOVIE_LENS, NORMAL_CUSTOM, COR_NORMAL_CUSTOM, SIFT_1M]
    ) and is_tradeoff:
        atoms -= np.mean(atoms + 1)
    signals = subset_2d(data, signal_idcs, sample_idcs)
    del data

    # add noise to data for epsilon-noise robustness experiments
    if add_noise:
        atoms = add_eps_noise(atoms)

    return atoms, signals


if __name__ == "__main__":
    atoms, queries = get_data(num_atoms=10000, len_signal=1000, data_type=GPT2_LM_HEAD)
    print(atoms.shape, queries.shape)
    print(atoms.dtype)
