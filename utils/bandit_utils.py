import numpy as np
import numba as nb

from utils.constants import HOEFFDING


@nb.njit
def get_ci(
    delta: np.ndarray,
    var_proxy: np.ndarray,  # Variance proxy of sub-gaussian r.v.
    num_samples: int,
    ci_bound: str = HOEFFDING,
    with_replacement: bool = True,
    pop_size: int = 10**10,
    sum_var_proxy: float = None,
) -> np.ndarray:
    if with_replacement:
        if ci_bound == HOEFFDING:  # see https://arxiv.org/pdf/2006.06856.pdf#page=20
            return np.sqrt(-2 * np.log(delta) * var_proxy / num_samples)
    else:
        if (
            ci_bound == HOEFFDING
        ):  # see Corollary 1 in https://arxiv.org/pdf/1812.06360.pdf
            rho = min(
                1 - (num_samples - 1) / pop_size,
                (1 - num_samples / pop_size) * (1 + 1 / num_samples),
            )
            if sum_var_proxy is not None:
                return np.sqrt(
                    -2 * rho * np.log(delta) * sum_var_proxy / num_samples**2
                )
            return np.sqrt(-2 * rho * np.log(delta) * var_proxy / num_samples)
