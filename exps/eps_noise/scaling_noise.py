import numpy as np
from exps.core_scaling.scaling_baselines import scaling_baselines
from utils.constants import (
    NETFLIX,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    SCALING_NUM_ATOMS,
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
    NOISE_VAR,
)


def scaling_noise(algorithms):
    """
    See if the scaling behavior for the MIPS algorithms is constant even for the noise-induced Netflix dataset.
    By default, the algorithm tested is BanditMIPS.
    """
    # Get data for the scaling experiments for the datasets (not the high-dimensional datasets)
    print(f"=> add noise {NOISE_VAR}")
    scaling_baselines(algorithms=algorithms, add_noise=True)


if __name__ == "__main__":
    scaling_noise([ACTION_ELIMINATION])