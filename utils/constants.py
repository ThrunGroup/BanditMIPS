NUM_STDS = 1  # A multiplier on the number of standard deviations we use when creating confidence intervals
BATCH_SIZE = 10
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Control/Independent Variables
NUMBER_OF_ATOMS = "NUMBER_OF_ATOMS"
DIMENSION_OF_ATOMS = "DIMENSION_OF_ATOM"

# MAB Algorithms
MEDIAN_ELIMINATION = "MEDIAN_ELIMINATION"
ACTION_ELIMINATION = "ACTION_ELIMINATION"
ADAPTIVE_ACTION_ELIMINATION = "ADAPTIVE_ACTION_ELIMINATION"
GREEDY_MIPS = "GREEDY_MIPS"
PCA_MIPS = "PCA_MIPS"
LSH_MIPS = "LSH_MIPS"
NEQ_MIPS = "NEQ_MIPS"
SYMMETRIC = "SYMMETRIC"
ASYMMETRIC = "ASYMMETRIC"
BUCKET_ACTION_ELIMINATION = "BUCKET_ACTION_ELIMINATION"
HNSW_MIPS = "HNSW_MIPS"
NAPG_MIPS = "NAPG_MIPS"
H2ALSH = "H2ALSH"
NAIVE = "NAIVE"

ALG_TO_COLOR = {
    MEDIAN_ELIMINATION: "#8c564b",
    ACTION_ELIMINATION: "#ff7f0e",
    ADAPTIVE_ACTION_ELIMINATION: "#d62728",
    GREEDY_MIPS: "#2ca02c",
    PCA_MIPS: "#9467bd",
    LSH_MIPS: "#1f77b4",
    NEQ_MIPS: "#808080",
    SYMMETRIC: "#e377c2",
    ASYMMETRIC: "#7f7f7f",
    HNSW_MIPS: "#F4D03F",
    NAPG_MIPS: "#F409FF",
    BUCKET_ACTION_ELIMINATION: "#17becf",
    H2ALSH: "#5d8694",
    NAIVE: '#497E76',
}

# Datasets
ADVERSARIAL_CUSTOM = "ADVERSARIAL_CUSTOM"
NORMAL_CUSTOM = "NORMAL_CUSTOM"
UNIFORM_PAPER = "UNIFORM_PAPER"
NORMAL_PAPER = "NORMAL_PAPER"
SYNTHETIC = "SYNTHETIC"
NETFLIX = "NETFLIX"
NETFLIX_TRANSPOSE = "TRANSPOSED_NETFLIX"
MOVIE_LENS = "MOVIE_LENS"
YAHOO = "YAHOO"
COR_NORMAL_CUSTOM = "CORRELATED_NORMAL_CUSTOM"
POSITIVE_COR_NORMAL_CUSTOM = "POSITIVE_COR_NORMAL_CUSTOM"
CRYPTO_PAIRS = "CRYPTO_PAIRS"
CLEAR_LEADER_HARD = "CLEAR_LEADER_HARD"
CLEAR_LEADER_SOFT = "CLEAR_LEADER_SOFT"
NO_CLEAR_LEADER = "NO_CLEAR_LEADER"
HIGHLY_SYMMETRIC = "HIGHLY_SYMMETRIC"
TOY_SINE = "TOY_SINE"
TOY_DISCRETE = "TOY_DISCRETE"
MNIST_T = "MNIST_T"
SIFT_1M = "SIFT_1M"
SIMPLE_SONG = "SIMPLE_SONG"
GPT2_LM_HEAD = "GPT2_LM_HEAD"
OPT_LM_HEAD = "OPT_LM_HEAD"

# Concentration Inequality bound
HOEFFDING = "HOEFFDING"

# Scaling Experiments

NUM_SEEDS = 30
BUCKET_AE_NUM_SEEDS = 5

DEFAULT_MAXMIN = (10, 0)
CRYPTO_PAIRS_MAXMIN = (10 ** 5, 0)
NORMALIZED_MAXMIN = (1, -1)
DEFAULT_GREEDY_BUDGET = 2e2
SCALING_NUM_EXPERIMENTS = 10  # corresponds to the number of points on the plot
SCALING_NUM_ATOMS_CRYPTO_PAIRS = 100


SCALING_NUM_ATOMS = 100
SCALING_NUM_SIGNALS = 3  # this is to get the confidence intervals
SCALING_DELTA = 0.03  # Only used for UNIFORM_PAPER
SCALING_EPSILON = 0.0  # Only used for UNIFORM_PAPER
SCALING_DELTA_REAL = 0.01  # This is actually the value used for every dataset except UNIFORM_PAPER
SCALING_EPSILON_REAL = 0.01  # This is actually the value used for every dataset except UNIFORM_PAPER
SCALING_DEPTH = 2
SCALING_TOPK = 5
SCALING_NUM_HFUNC = 3
SCALING_NUM_TABLES = 5
SCALING_SIZE_MINMAX = (
    10 ** 4, 10 ** 6,
)
SCALING_H2ALSH_DELTA = 0.1
SCALING_H2ALSH_C0 = 1.5
SCALING_H2ALSH_C = 0.8
SCALING_H2ALSH_N0 = 200
SCALING_LOG_BUFFER = 1/1e6


SCAlING_SIZE_MINMAX_MOVIE = (   # movie dataset is only 6000 dimensions
    450, 5000,
)
SCAlING_SIZE_MINMAX_CRYPTO_PAIRS = (  # crypto pairs dataset has 1M dimensions
    100, 10 ** 6,
)
SCALING_H2ALSH_PARAMS = (0.4, 1.5, 0.8, 200) # it records (delta, c0, c, N0)

# Precision-Speed Tradeoff Experiments
TRADEOFF_NUM_EXPERIMENTS = 24  # corresponds to the number of points on the plot
TRADEOFF_NUM_ATOMS = 1000
TRADEOFF_DIMENSION = 10000
TRADEOFF_DIMENSION_MOVIE = 5000  # movie dataset is only 6000 dimensions
TRADEOFF_NUM_SIGNALS = 3
TRADEOFF_TOPK = 10

# Normal Custom dataset params
TRADEOFF_DEFAULT_EPSILON = 0.0
TRADEOFF_MAXMIN = (4.0, -4.0)
NORMAL_VAR = 1
DELTA_BINS = 4
DELTA_MINMAX = (0.0, 0.99)
DELTA_BINS_ACTION = 3
DELTA_BINS_REAL = 3
DELTA_MEDIAN = 0.5
EPSILON_MINMAX_ACTION = (0.0, 1.5)
EPSILON_MINMAX_MEDIAN = (0.0, 3.0)
GREEDY_BUDGET_BY = 5
MOVIE_LENS_MAXSIZE = 5000

GREEDY_BUDGET_MINMAX = (10, 50)
TRADEOFF_PCA_NUM_EXPERIMENTS = 4  # this is consistent with the paper
TRADEOFF_DEPTH_MINMAX = (0, 4)

TRADEOFF_LSH_NUM_EXPERIMENTS = 24  # this is consistent with the paper
TRADEOFF_HFUNC_MINMAX = (1, 10)
TRADEOFF_TABLES_MINMAX = (10, 1)

SYMMETRIC = "SYMMETRIC"
ASYMMETRIC = "ASYMMETRIC"

# Adversarial dataset params
DELTA_BINS_AD = 2
MAXMIN_AD = (0.5, -0.5)

# Greedy MIPS Experiment
GREEDY_ATOM_SIZE = [17, 18, 19]
GREEDY_QUERY_SIZE = [2, 5, 7]

# Real Data params
NUM_ATOMS_REAL = 1000
LEN_SIGNAL_NETFLIX = 10000
LEN_SIGNAL_MOVIE = 5000

MAX_SPEEDUP = 20
TRADEOFF_DELTA_MINMAX_REAL = (0, 0.5)
TRADEOFF_MAXMIN_REAL = (0.3, -0.3)
TRADEOFF_VAR_REAL = 2
TRADEOFF_DEPTH_MINMAX_REAL = (0, 10)
TRADEOFF_BATCH_SIZE_REAL = 5
DELTA_POWER_MINMAX = (0.05, 15)

# Crypto Pairs Data params
TRADEOFF_MAXMIN_CRYPTO_PAIRS = (10 ** 6, 0)
CRYPTO_PAIRS_SIGNAL_LENGTH = 10 ** 6

# H2ALSH TRADEOFF PARAMS
TRADEOFF_H2ALSH = {"powers_delta": (-3, 0), "c0": (1.2, 5), "c": (0.9, 0.2), "N0": (10, 10)}

# BUCKET ACTION ELIMINATION TRADEOFF PARAMS
TRADEOFF_BUCKET_ACTION_ELIMINATION = {"deltas": (10 ** -10, 0.99), "epsilons": (10 ** -10, 3), "bucket_num_samples": (1000, 10)}

# NEQ TRADEOFF PARAMS
TRADEOFF_CODEBOOK_MINMAX = (100, 1)
TRADEOFF_CODEWORD_MINMAX = (100, 1)

# Large Scale params
LARGE_SIZE_MINMAX = (10, 5 * 10 ** 6)
LARGE_NUM_ATOMS = 100
LARGE_NUM_EXPERIMENTS = 20
LARGE_NUM_SEEDS = 10
LARGE_DELTA = 0.01
LARGE_EPSILON = 0.01
LARGE_MAXMIN = (0.5, -0.5)

# Caching params
CACHING_BUDGET = 0.01
TEMP_SIZE = 1  # this is to make caching compatible with numba

RECON_ERROR_THRESHOLD = 0.06

# For song dataset
SECONDS_PER_MINUTE = 60
SAMPLE_RATE = 44100

# experiment configs
SCALING_BASELINES_ALGORITHMS = [
    # GREEDY_MIPS,  # greedy approach
    # LSH_MIPS,  # mapping to higher dimension (Todo: include H2LSH after vectorization?)
    # PCA_MIPS,
    # NEQ_MIPS,
    # NAPG_MIPS,
    # MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    # ADAPTIVE_ACTION_ELIMINATION,  # bandit approach
]
SCALING_BASELINES_DATATYPES = [
    # NORMAL_CUSTOM,
    # COR_NORMAL_CUSTOM,
    #UNIFORM_PAPER,
    #CLEAR_LEADER_HARD,
    NETFLIX,
    # MOVIE_LENS,
    # NORMAL_PAPER,
]

# eps-noise robustness
NOISE_VAR = 0.2