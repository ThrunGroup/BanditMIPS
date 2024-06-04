from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_NUM_ATOMS,
    NEQ_MIPS,

    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,

    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
)

from neq_mips.constants import TRADEOFF_NEQ_NUM_EXPERIMENTS


if __name__ == '__main__':
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        if data_type is NETFLIX:
            num_atoms = NUM_ATOMS_REAL
            len_signals = LEN_SIGNAL_NETFLIX
        elif data_type is MOVIE_LENS:
            num_atoms = NUM_ATOMS_REAL
            len_signals = LEN_SIGNAL_MOVIE
        else:
            num_atoms = TRADEOFF_NUM_ATOMS
            len_signals = TRADEOFF_DIMENSION

        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NEQ_NUM_EXPERIMENTS,
            num_signals=len_signals,
            data_type=data_type,
            mips_alg=NEQ_MIPS,
        )