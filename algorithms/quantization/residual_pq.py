from __future__ import division
from __future__ import print_function
import numpy as np
import math
from scipy.cluster.vq import vq, kmeans2

from neq_mips.constants import *


class ResidualPQ(object):
    """
    This class has the same member functions as the PQ class and receives a list of PQs as input. Thus, all functions
    call each PQ object's identically named functions. This means encode/decode have an additional dimension
    for each PQ in the input list.
    """
    def __init__(self, pqs=None, verbose=False):
        assert len(pqs) > 0
        self.verbose = verbose
        self.deep = len(pqs)
        self.code_dtype = pqs[0].code_dtype
        self.M = max([pq.M for pq in pqs])
        self.pqs = pqs

        for pq in self.pqs:
            assert pq.code_dtype == self.code_dtype

    def class_message(self):
        messages = ""
        for i, pq in enumerate(self.pqs):
            messages += pq.class_message()
        return messages

    def fit(self, T, iter, D=None):
        assert T.dtype == np.float32
        assert T.ndim == 2

        vecs = np.empty(shape=T.shape, dtype=T.dtype)
        vecs[:, :] = T[:, :]
        if D is not None:
            vecs_d = np.empty(shape=D.shape, dtype=D.dtype)
            vecs_d[:, :] = D[:, :]

        # training each subspace quantizer
        for layer, pq in enumerate(self.pqs):
            pq.fit(vecs, iter)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed    # fitting on the residuals
            del compressed

            if D is not None:
                compressed_d = pq.compress(vecs_d)
                vecs_d -= compressed_d

            if self.verbose:
                norms = np.linalg.norm(vecs, axis=1)
                print("# layer: {},  residual average norm : {} max norm: {} min norm: {}"
                      .format(layer, np.mean(norms), np.max(norms), np.min(norms)))

        return self

    def encode(self, vecs):
        codes = np.zeros((len(vecs), self.deep, self.M), dtype=self.code_dtype)  # N * deep * M
        for i, pq in enumerate(self.pqs):
            codes[:, i, :pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(codes[:, i, :pq.M])
        return codes  # N * deep * M

    def decode(self, codes):
        vecss = [pq.decode(codes[:, i, :pq.M]) for i, pq in enumerate(self.pqs)]
        return np.sum(vecss, axis=0)

    def compress(self, X):
        N, D = np.shape(X)

        sum_residual = np.zeros((N, D), dtype=X.dtype)

        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        for i, pq in enumerate(self.pqs):
            compressed = pq.compress(vecs)
            vecs[:, :] = vecs - compressed
            sum_residual[:, :] = sum_residual + compressed
            del compressed
        return sum_residual