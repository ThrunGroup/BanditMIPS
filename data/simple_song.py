import numpy as np
import soundfile as sf

from algorithms.action_elimination import action_elimination
from utils.constants import (
    SAMPLE_RATE,
    SECONDS_PER_MINUTE,
)


def create_sin_wave(
    frequency_multiplier: float,
    sampling_frequency: int = SAMPLE_RATE,
    num_samples: int = SAMPLE_RATE * SECONDS_PER_MINUTE,
):
    """
    Create a discretized sine wave with the ith value being sin(frequency_multiplier * i / sampling_frequency).
    Sampling frequency should be given in Hz.

    :param frequency_multiplier: Frequency multiplier
    :param sampling_frequency: Sampling frequency
    :param num_samples: Number of samples
    :return: Array of sine wave values
    """
    assert frequency_multiplier > 0, "Not implemented for <= 0 frequencies yet"
    phase = frequency_multiplier * np.arange(num_samples) / sampling_frequency
    return np.sin(phase)


def create_cos_wave(
    frequency_multiplier: float,
    sampling_frequency: int = SAMPLE_RATE,
    num_samples: int = SAMPLE_RATE * SECONDS_PER_MINUTE,
):
    """
    Create a discretized cosine wave with the ith value being cos(frequency_multiplier * i / sampling_frequency).
    Sampling frequency should be given in Hz.

    :param frequency_multiplier: Frequency multiplier
    :param sampling_frequency: Sampling frequency
    :param num_samples: Number of samples
    :return: Array of cosine wave values
    """
    assert frequency_multiplier > 0, "Not implemented for <= 0 frequencies yet"
    phase = frequency_multiplier * np.arange(num_samples) / sampling_frequency
    return np.cos(phase)


def stack(arr_: np.ndarray, num: int):
    """
    Convenience function to horizontally stack an array (increase its width) by num times.

    :param arr_: Array to stack horizontally
    :param num: Number of times to duplicate the array
    :return: Horizontally replicated array
    """
    return np.hstack([arr_ for _ in range(num)])


def create_note_waves(num_minutes: int = 10):
    """
    Create the sine waves for the notes C0, E0, G0, C1, E1, G1.

    :return: tuple of numpy arrays, one array per note
    """
    # Note: Need to use 2*\pi exactly here! Using 6.28 means that the waves won't go through
    # full cycles each second, which means stacking them 10 times will give different answers from just continuing them
    two_pi = 2 * np.pi
    C0 = create_sin_wave(
        frequency_multiplier=256 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )
    E0 = create_sin_wave(
        frequency_multiplier=330 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )
    G0 = create_sin_wave(
        frequency_multiplier=392 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )

    C1 = create_sin_wave(
        frequency_multiplier=2 * 256 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )
    E1 = create_sin_wave(
        frequency_multiplier=2 * 330 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )
    G1 = create_sin_wave(
        frequency_multiplier=2 * 392 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_minutes,
    )
    return C0, E0, G0, C1, E1, G1


def create_song(num_minutes: int = 10):
    """
    Creates a simple song of length num_minutes minutes.
    The song is created by playing C0 + 2*E0 + 3*G0 for 1 minute, then 3*G0 + 2.5*C1 + 1.5*E1 for 1 minute, and
    repeatedly alternating.

    The default parameters make this 44100 * 60 * 10 = 26,460,000 dimensional.

    :param num_minutes: Number of minutes the song should play
    :return:
    """
    C0, E0, G0, C1, E1, G1 = create_note_waves(num_minutes=1)
    song = np.array([])
    assert int(num_minutes) == num_minutes, "Must pass integer number of num_minutes"
    for i in range(int(num_minutes)):
        if i % 2 == 0:
            song = np.hstack([song, C0 + 2 * E0 + 3 * G0])
        else:
            song = np.hstack([song, 3 * G0 + 2.5 * C1 + 1.5 * E1])

    return song


def main():
    num_mins = 10
    song = create_song(num_minutes=num_mins)
    signals = song.reshape(1, -1)
    # Note: Need to use 2*\pi exactly here! Using 6.28 means that the waves won't go through
    # full cycles each second, which means stacking them 10 times will give different answers from just continuing them
    two_pi = 2 * np.pi
    frequencies = two_pi * np.array([256, 330, 392, 2 * 256, 2 * 330, 2 * 392])
    atoms = np.array(
        [
            create_sin_wave(
                frequency_multiplier=f,
                sampling_frequency=SAMPLE_RATE,
                num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_mins,
            )
            for f in frequencies
        ]
    )
    atom1, budget1 = action_elimination(
        atoms, signals, var_proxy=0.1, epsilon=0, delta=0.0003, abs=True, batch_size=300
    )
    print(atom1, budget1)
    assert atom1[0, 0] == 2, "Wrong fourier component identified"

    atom4, budget4 = action_elimination(
        stack(atoms, 4),
        stack(signals, 4),
        var_proxy=0.1,
        epsilon=0,
        delta=0.0003,
        abs=True,
        batch_size=300,
    )
    assert atom4[0, 0] == 2, "Fourier component does not agree when duplicating data 4x"
    assert budget4 == budget1, "Budget does not agree when duplicating data 4x"

    G0 = create_sin_wave(
        frequency_multiplier=392 * two_pi,
        sampling_frequency=SAMPLE_RATE,
        num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * num_mins,
    )
    signal_G0_removed = signals - 3 * G0

    atom_g0, budget_g0 = action_elimination(
        atoms,
        signal_G0_removed,
        var_proxy=0.1,
        epsilon=0,
        delta=0.001,
        abs=True,
        batch_size=300,
    )
    print(atom_g0, budget_g0)
    assert atom_g0[0, 0] == 3, "Wrong fourier component identified in second step"


if __name__ == "__main__":
    main()
