import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter
from typing import Tuple, List

from utils.lsh_utils import *
from data.custom_data import generate_custom_data
from utils.constants import (
    SYMMETRIC,
    ASYMMETRIC,
    SCALING_TOPK,
    SCALING_NUM_HFUNC,
    SCALING_NUM_TABLES
)


# for debugging purposes
np.random.seed(0)


class LSH_MIPS:
    """
    Class for solving MIPS using LSH. The main logic divides into preprocessing the data and
    querying the signal vector. Used https://github.com/jfpu/lsh_mips as reference for implementation.
    The basic logic of running LSH_MIPS is as follows:
        1. initialize LSH_MIPS class with all it's parameters
        2. call prepare_data() method and create_lsh_table()
        3. call run()

    :type: either symmetric or asymmetric
    :data: data used to build hash indices
    :queries: 1d array of query data
    :rand_range: random range for normalization
    :m: the number of dimensions to extend the dataset to convert from MIPS to NN search problem.
    """

    def __init__(
        self,
        m: int = 3,
        rand_range: int = 1.0,
        num_hfunc: int = 5,
        num_tables: int = 10,
        type: str = ASYMMETRIC,
    ) -> None:
        """
        :m: the number of dimensions to extend the data, query by
        :rand_range: the range for the constant factor of the hash function
        :num_hfunc: Number of hyperplanes (hash value is an AND construction of hyperplanes)
        :num_tables: Number of hash tables (hash value is an OR construction of tables)
        :type: either asymmetric or symmetric
        """
        if type == SYMMETRIC:
            self.m = 1  # m is always 1 if we're doing symmetric lsh mips
            self.hash_func = lambda data, hyperplanes: np.matmul(data, hyperplanes)

        elif type == ASYMMETRIC:
            self.m = m
            b = np.random.uniform(0, rand_range)  # 0 < b < rand_range
            self.hash_func = (
                lambda data, hyperplanes: (np.matmul(data, hyperplanes) + b)
                / rand_range
            )
        else:
            raise Exception("Invalid LSH type declared")

        self.type = type
        self.num_hfunc = num_hfunc
        self.num_tables = num_tables

    def preprocess_data(
        self,
        atoms: np.ndarray
    ) -> None:
        """
        Preprocess the data before hashing into buckets. Preprocessing includes these steps:
            1. scaling transformation of data (to convert from MIPS to NNS)
            2. vector concatenation corresponding to P(.) and Q(.) of the data (symmetric lsh only has P(.))
        """
        # data & query transformation. norm_data is S(x) in the original paper
        _, _, norm_data = g_transformation(atoms)

        # expand k dimension into k+2m dimension following directions in original paper
        if self.type == SYMMETRIC:
            self.ext_data = np.array(g_index_simple_extend(norm_data))
        else:
            self.ext_data = np.array(g_index_extend(norm_data, self.m))

        self.d = len(norm_data[0]) + self.m
        assert self.d == len(self.ext_data[0])  # num dimensions

        # hash the preprocessed data into the corresponding buckets
        self.create_lsh_table(self.ext_data)

    def preprocess_query(
        self,
        signal: np.ndarray
    ) -> None:
        """
        Preprocess signal to turn into NN search problem.

        :signal: the query to be preprocessed
        :returns: None
        """
        norm_query = g_normalization(signal)
        if self.type == SYMMETRIC:
            self.ext_query = np.array(g_query_simple_extend(norm_query))
        else:
            self.ext_query = np.array(g_query_extend(norm_query, self.m))

        assert len(norm_query) + self.m == len(self.ext_query)  # num dimensions

    def create_lsh_table(self, data):
        """
        Hash the dataset into it's corresponding buckets. Each element in the bucket is the "index" of the dataset row.

        :data: the preprocessed dataset
        """
        assert (
            data.shape[1] == self.d
        )  # make sure that we're using the preprocessed data

        # create num_tables hash tables where each table consists of num_hfunc hyperplanes
        self.hash_tables = []
        for i in range(self.num_tables):
            if self.num_hfunc == 0:
                hyperplanes = np.zeros((1, self.d))  # map to single bucket
            else:
                hyperplanes = np.random.normal(0.0, 1.0, (self.num_hfunc, self.d))

            self.hash_tables.append(
                # single element of table looks like (hyperplanes, dict(hash_value: {idcs_of_data}))
                (hyperplanes, defaultdict(lambda: []))
            )

        # hash the values
        for hyperplanes, hash_dict in self.hash_tables:
            # standard amplification procedure (a hash value is an array).
            # The final number of buckets is 2^num_hfunc
            values = self.hash_func(data, hyperplanes.T)
            hash_values = np.where(values > 0, 1, 0)

            for i, hash_value in enumerate(hash_values):
                bucket = binary_to_num(hash_value)
                hash_dict[bucket].append(i)

    def run(self, top_k=SCALING_TOPK):
        """
        Run the nearest neighbor search to find the specified number of neighbors (default 1).

        :top_k: the number of top candidates you want to find
        """
        # make sure we're using the preprocessed data and queries
        assert self.ext_data.shape[1] == self.d
        assert len(self.ext_query) == self.d

        budget = 0
        candidates = set()  # this prevents duplicates
        for hyperplanes, hash_dict in self.hash_tables:
            value = self.hash_func(hyperplanes, self.ext_query)
            hash_value = np.where(value > 0, 1, 0)

            # get matches
            bucket = binary_to_num(hash_value)
            matches = hash_dict.get(bucket, [])  # return [] if match doesn't exist
            candidates.update(matches)

            # dot product with one hyperplane costs self.d budget
            h, d = hyperplanes.shape
            budget += h * d

        # return budgets and candidates matched
        candidates = np.asarray(list(candidates))

        # There aren't enough candidates. Still return the ones that we have
        if len(candidates) < top_k:
            #print(f"have {len(candidates)} collisions but want {top_k}!")
            top_k = len(candidates)

            if top_k == 0:
                return [], budget

        # NN search for the best top_k candidates. Need to check all candidates at least once to compare with query.
        # and one more time since the complexity of np.argpartition is O(n) (uses a heap).
        dist_vec = np.linalg.norm(self.ext_data[candidates] - self.ext_query, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)
        budget += (len(candidates) + 1) * self.d
        return candidates[idx[:top_k]], budget


def run_lsh_mips(
    atoms: np.ndarray,
    signals: np.ndarray,
    num_best_atoms: int = SCALING_TOPK,
    num_hfunc: int = SCALING_NUM_HFUNC,
    num_tables: int = SCALING_NUM_TABLES,
    type: str = ASYMMETRIC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LSH mips to find num_best_atoms candidates for the atoms and signals inputted.

    :param atoms: Atoms array
    :param signals: Signals array
    :param num_best_atoms: Number of the best atoms we choose
    :param num_hfunc: Number of hyperplanes (hash value is an AND construction of hyperplanes)
    :param num_tables: Number of hash tables (hash value is an OR construction of tables)
    :return: An array of final answers for MIPS, and array of budgets (number of computations).
    """
    lsh_object = LSH_MIPS(
        num_hfunc=num_hfunc,
        num_tables=num_tables,
        type=type
    )
    lsh_object.preprocess_data(atoms)

    candidates_array = []
    budgets_array = []
    for signal in signals:
        lsh_object.preprocess_query(signal)
        candidates, budgets = lsh_object.run(num_best_atoms)
        candidates_array.append(candidates)
        budgets_array.append(budgets)

    return np.array(candidates_array, dtype=object), np.array(budgets_array)


if __name__ == "__main__":
    # This is just for @Mo to test the PR and verify that the lsh-mips object runs.
    # To compare with the naive mips, check branch test_baselines which has a function called compare_to_naive
    seed = 0
    top_k = 5
    atoms, signal = generate_custom_data(num_atoms=100, len_signal=1000000, seed=seed, data_type="NORMAL_PAPER")
    signal = signal[0]
    import time

    start_time = time.time()
    LSHMIPS = LSH_MIPS(atoms, signal)
    LSHMIPS.prepare_data()
    LSHMIPS.create_lsh_table(LSHMIPS.ext_data)
    print(f"Preprocess: {time.time() - start_time}(s)")

    start_time = time.time()
    budget, candidates = LSHMIPS.run(LSHMIPS.ext_data, LSHMIPS.ext_query, top_k=top_k)
    print(f"lsh_mips: {time.time() - start_time}(s)")

    start_time = time.time()
    print(f"accuracy: ", len(np.intersect1d(candidates, np.argsort(-atoms @ signal)[:10])) / top_k)
    print(f"naive: {time.time() - start_time}(s)")

    print(f" =>> Top {top_k} candidate(s): ", candidates)
    print(" =>> budget: ", budget)
