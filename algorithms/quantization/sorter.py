import numpy as np
import numba as nb
import math


@nb.jit
def true_positives(ranks, Q, G, top_k):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        # get rid of multiplication of "len(Q[0])/t" for recall
        result[i] = len(
            np.intersect1d(G[i], ranks[i][:(top_k + 1)])
        ) / (len(Q) * top_k)
    return result


@nb.jit
def parallel_sort(compressed, Q, top_k, num_codebooks, num_codewords):
    """
    For each q in 'Q', sort the items in "compressed" by their euclidean product distance.
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return: indices of the topk closest items and the budget
    """
    budgets = np.zeros(len(Q))
    ranks = np.empty((Q.shape[0], min(top_k, compressed.shape[0]-1)), dtype=np.int32)
    for i in nb.prange(Q.shape[0]):
        distances = compressed @ -Q[i]
        budgets[i] += (num_codewords * len(Q[i]) + len(distances) * num_codebooks)

        # originally the method is to argpartition -> argsort, but numba doesn't support argpartition
        size = min(top_k, len(distances) - 1)
        ranks[i, :] = np.argsort(distances)[:size]
        budgets[i] += (len(distances) + size * math.log(size))
    return ranks, budgets


class Sorter(object):
    def __init__(self, compressed, Q, top_k, num_codebooks, num_codewords):
        self.Q = Q
        self.ranks, budgets = parallel_sort(compressed, Q, top_k, num_codebooks, num_codewords)
        self.budgets = budgets

    def get_candidates(self):
        return self.ranks[:]

    def precision(self, G, T):
        t = min(T, len(self.ranks[0]))
        return t, self.avg_precision(G, T)

    def avg_precision(self, G, top_k):
        assert len(self.Q) == len(self.ranks), "number of query not equals"
        assert len(self.ranks) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        return true_positives(self.ranks, self.Q, G, top_k)