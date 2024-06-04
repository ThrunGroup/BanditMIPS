import numpy as np

from algorithms.quantization.norm_pq import NormPQ
from algorithms.quantization.residual_pq import ResidualPQ
from algorithms.quantization.base_pq import PQ
from exps.scaling_exp import scaling_exp
from utils.constants import (
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    NEQ_MIPS,

    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS
)


if __name__ == '__main__':
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        scaling_exp(
            num_atoms=SCALING_NUM_ATOMS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=NEQ_MIPS,
        )