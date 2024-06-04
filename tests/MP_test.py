import numpy as np

from algorithms.matching_pursuit import orthogonal_matching_pursuit
from utils.utils import get_recon_error
from data.custom_data import create_toy_waves, create_toy_discrete
from utils.constants import (
    ACTION_ELIMINATION,
    RECON_ERROR_THRESHOLD,
)


def run_toy_dataset(
        name="discrete",
        num_atoms=100,
        mul=3000,
        verbose=False,
):
    """
    Runs bandit-OMP on the specified toy dataset. The objective is for the OMP to find the first, second, third
    atom indices, in that order
    """
    if verbose:
        print(f"number of atoms: {num_atoms}")
        print(f"dimension of query: {mul * 3}")
    if name == "discrete":
        if verbose:
            print("\n-------- these are the results for the toy discrete --------")
        atoms, query = create_toy_discrete(num_atoms, mul)
    else:
        if verbose:
            print("\n-------- these are the results for the sine toy --------")
        atoms, query = create_toy_waves(num_atoms, mul * 3)

    candidates, gamma, budget = orthogonal_matching_pursuit(
        atoms=atoms.T,
        signal=query,
        n_nonzero_coefs=3,
        min_residual=0.01,
        bandit_alg=ACTION_ELIMINATION,
        abs=True,
        seed=0,
        use_cache=False,
    )
    assert len(np.intersect1d(candidates, np.array([0, 1, 2]))) == len(candidates), \
        "Need to find the indices [0, 1, 2]"
    assert get_recon_error(gamma, atoms[candidates], query) < RECON_ERROR_THRESHOLD / 100, \
        f"Reconstruction error must be less than {RECON_ERROR_THRESHOLD / 100}"
    print(f"passes all tests for {name} dataset")


if __name__ == "__main__":
    run_toy_dataset(name="discrete")
    run_toy_dataset(name="waves")