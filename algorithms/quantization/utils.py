from multiprocessing import cpu_count
import numpy as np
import numba as nb
import math

from algorithms.quantization.sorter_batch import BatchSorter
from neq_mips.constants import (
    NUM_CODEWORDS,
    NUM_CODEBOOKS,
    CHUNK_SIZE,
    TRAINING_ITERATION,
    BATCH_SIZE,
    NEQ_DEFAULT_TOPK,
    TRAIN_SIZE
)


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)  # divide by zero problem
    return norms, normalized_vecs


def chunk_compress(pq, vecs):
    """
    Run pq's .compress function on the vecs in batches to get the codebook approximation. Assumes that pq is fitted.
    By default, the batchsize is the same size as the training set (i.e. compress all at once).
    :param pq: the product quantization object
    :param vecs: analogous to the atoms
    """
    chunk_size = CHUNK_SIZE    # same size as training set by default
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in range(math.ceil(len(vecs) / chunk_size)):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def execute(
    seed: int,
    pq: object,
    X: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    num_codebooks: int = NUM_CODEBOOKS,
    num_codewords: int = NUM_CODEWORDS,
    top_k=NEQ_DEFAULT_TOPK,
    train_size=TRAIN_SIZE
):
    """
    Function that preprocesses and queries the dataset following the NEQ logic.
    :params pq: the product quantizer
    :params X: analogous to the atoms
    :params Q: analogous to signals
    :params G: analogous to the naive candidates array
    """
    # preprocessing (i.e. creating the codebooks)
    np.random.seed(seed)
    pq.fit(X[:train_size].astype(dtype=np.float32), iter=TRAINING_ITERATION)
    compressed = chunk_compress(pq, X)
    candidates, budgets = BatchSorter(
        compressed=compressed,
        Q=Q,
        G=G,
        top_k=top_k,
        num_codebooks=num_codebooks,
        num_codewords=num_codewords,
        batch_size=BATCH_SIZE
    ).candidate()
    return candidates, budgets