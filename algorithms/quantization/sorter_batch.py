import numpy as np
import math

from algorithms.quantization.sorter import Sorter


class BatchSorter(object):
    def __init__(self, compressed, Q, G, top_k, num_codebooks, num_codewords, batch_size, get_precision=False):
        self.Q = Q
        self.candidates = np.zeros((len(Q), top_k))
        self.precisions = np.zeros(len(Q))
        self.budgets = np.zeros(len(Q))

        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = Sorter(compressed, q, top_k, num_codebooks, num_codewords)
            self.budgets[i*batch_size: (i + 1) * batch_size] += sorter.budgets
            #import ipdb; ipdb.set_trace()
            self.candidates[:, i*batch_size: (i + 1) * batch_size] += sorter.get_candidates()
            if get_precision:
                self.precisions[i * batch_size: (i + 1) * batch_size] += sorter.avg_precision(g, top_k)

    def candidate(self):
        return self.candidates, self.budgets

    def precision(self):
        return self.precisions, self.budgets
