from __future__ import division
from __future__ import print_function
import numpy as np
import math
from scipy.cluster.vq import vq, kmeans2
from algorithms.quantization.utils import normalize

from neq_mips.constants import *


class NormPQ(object):
    """
    Receives ResidualPQ object since that it what's going to be used to train the direction and norm codebooks.
    The member functions are as follows:
        - fit: trains the same number of direction and norm codebooks following Algorithm 2 from the paper
        - {encode/decode}_norm: encodes/decodes the norms with the norm codebooks
        - compress: return the results of Algorithm 1 from paper.
    """
    def __init__(self, n_percentile, quantize, true_norm=False, verbose=False):
        # n_percentile -> number of codewords for norm training (default same as Ks)
        self.M = PARTITION
        self.n_percentile, self.true_norm, self.verbose = n_percentile, true_norm, verbose
        self.code_dtype = np.uint8 if n_percentile <= 2 ** 8 \
            else (np.uint16 if n_percentile <= 2 ** 16 else np.uint32)

        self.percentiles = None  # These are the norm codebooks. The direction codebooks exist in the RQ object.
        self.quantize = quantize

    def class_message(self):
        return "NormPQ, percentiles: {}, quantize: {}".format(self.n_percentile, self.quantize.class_message())

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.n_percentile <= N, "the number of norm intervals shouldn't be less than Ks"

        norms, normalized_vecs = normalize(vecs)  # get the true norm
        self.quantize.fit(normalized_vecs, iter)  # fit on direction vector (i.e. x' in paper)

        # get the approximation for the relative norm (i.e. l_x in paper)
        compressed_vecs = self.quantize.compress(normalized_vecs)
        temp = np.linalg.norm(compressed_vecs, axis=1)
        norms = norms / np.where(temp < NORM_BUFFER, NORM_BUFFER, temp)
        self.percentiles, _ = kmeans2(norms[:], self.n_percentile, iter=iter, minit='points')
        return self

    def encode_norm(self, norms):
        norm_index, _ = vq(norms[:], self.percentiles)
        return norm_index

    def decode_norm(self, norm_index):
        return self.percentiles[norm_index]

    def compress(self, vecs):
        norms, normalized_vecs = normalize(vecs)
        compressed_vecs = self.quantize.compress(normalized_vecs)   # approximation of direction vectors
        del normalized_vecs

        norms = norms / np.linalg.norm(compressed_vecs, axis=1)
        return (compressed_vecs.transpose() * norms).transpose()