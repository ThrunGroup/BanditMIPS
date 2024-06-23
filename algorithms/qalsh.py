import numpy as np
import numba as nb
from math import log, sqrt, ceil, log2
from scipy import stats
from typing import Tuple


class QALSH:
    """
    Python Implementation of Query-aware Asymmetric Locally Sensitive Hashing Algorithm for
    MIPS(Maximum Inner Product Search) Problem, see https://github.com/HuangQiang/QALSH
    """

    def __init__(
        self, atoms: np.ndarray, delta: float = 0.1, approx_const: float = 2.0,
    ):
        """
        :param atoms: 2d numpy array of atoms
        :param delta: Error probability
        :param approx_const: Approximation constant that decides the radius of each hash bucket.
        """
        assert 0 < delta < 1, "Error probability should be strictly between 0 and 1"
        assert (
            approx_const > 1
        ), "Approximation constant(c) in c-ANN search should be greater than 1"
        self.atoms = atoms
        self.random_vectors = None
        width = sqrt(
            (8 * approx_const ** 2 * log(approx_const)) / (approx_const ** 2 - 1)
        )  # See Lemma 5
        beta = min(
            100 / atoms.shape[0], 0.01
        )  # Beta denotes collision threshold percentage, see 5.3 in the paper

        # To see what p1 and p2 denotes and how they are calculated, see 3.2 in the paper
        p1 = 1 - 2 * stats.norm.cdf(-width / 2)
        p2 = 1 - 2 * stats.norm.cdf(-width / (2 * approx_const))
        eta = sqrt(log(2 / beta) / log(1 / delta))
        alpha = (eta * p1 + p2) / (1 + eta)  # See 5.3
        num_hash_tables = int(
            (sqrt(log(2 / beta)) + sqrt(log(1 / delta))) ** 2 / (2 * (p1 - p2) ** 2)
        )

        self.m = min(num_hash_tables, len(atoms))  
        self.delta = delta  # Error probability: QALSH algorithm solves c-ANN search with 1/2 - delta probability
        self.approx_const = approx_const  # The approximation constant c in c-ANN search
        self.num_fp = min(int(beta * atoms.shape[0]), 1)  # Number of false positives allowed
        self.collision_threshold = int(alpha * self.m)
        self.width = width  # Width of hash bucket
        self.is_generate_hash_table = False

    def generate_hash_tables(self) -> int:
        """
        Generate hash tables with a query-aware hash function described in the 3.1 of the paper.

        :return: Sample complexity of generating hash tables
        """
        if self.is_generate_hash_table:
            print("Hash table is already generated")
            return 0

        self.random_vectors = np.random.normal(size=(self.atoms.shape[1], self.m))
        self.hash_tables = [None] * self.m
        for i in range(self.m):
            hash_values = self.atoms @ self.random_vectors[:, i]
            atoms_sort_idcs = np.argsort(
                hash_values
            )  # Sorting indices of atoms in terms of their hash values
            self.hash_tables[i] = [atoms_sort_idcs, hash_values[atoms_sort_idcs]]
        self.is_generate_hash_table = True

        # Cost in generating random vectors + having dot product between atoms and random vectors + sorting hash tables
        sample_complexity = (
            self.m * self.atoms.shape[1]
            + self.m * self.atoms.shape[0] * self.atoms.shape[1]
            + self.m * self.atoms.shape[0] * log(self.atoms.shape[0])
        )
        return int(sample_complexity)

    def nn_search(self, query: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, int]:
        """
        Do Nearest Neighbor search on given atoms and query after narrowing down the possible candidates by
        removing atoms that don't collide much with query in the previously generated hash tables.

        :param query: 1d numpy array of query vector
        :param top_k: Number of atoms that H2ALSH algorithm identify as top/best atoms in terms of their inner
        product values
        :return: top_k atoms and sample complexity
        """
        assert len(query.shape) == 1, "Query should be a 1d vector"
        assert self.is_generate_hash_table, "Hash tables should be generated"
        radius = 1
        candidates = np.array(
            [], dtype=np.int64
        )  # Indices of nn(nearest neighbor) candidates
        candidates_dist = np.array([])  # Distances between candidates and query
        collision_counts = np.zeros(self.atoms.shape[0])
        sample_complexity = 0
        k = 0  # Constant for radius update
        while len(candidates) <= self.num_fp + top_k - 1:
            nearest_atoms_dist = (
                []
            )  # Distance between the query and atoms that are nearest to the query in m hash
            # tables.
            for i in range(self.m):
                query_hash_value = np.dot(self.random_vectors[:, i], query)
                lower_bound = query_hash_value - 1 / 2 * self.width * radius
                upper_bound = query_hash_value + 1 / 2 * self.width * radius
                lower_idx = np.searchsorted(self.hash_tables[i][1], lower_bound)
                upper_idx = np.searchsorted(self.hash_tables[i][1], upper_bound) - 1
                lower_idx = min(lower_idx, self.atoms.shape[0] - 1)
                upper_idx = max(upper_idx, 0)

                # Append the distance between the query and the atom that is nearest to query in the hash table. This
                # is to skip unnecessary rounds. See 4.2 for the detailed explanation.
                nearest_atoms_dist.append(
                    min(
                        abs(self.hash_tables[i][1][lower_idx] - query_hash_value),
                        abs(self.hash_tables[i][1][upper_idx] - query_hash_value),
                    )
                )
                idcs_atoms_collided = self.hash_tables[i][0][lower_idx:upper_idx]
                collision_counts[idcs_atoms_collided] += 1
            sample_complexity += self.m * self.atoms.shape[1] + self.m * log2(
                self.atoms.shape[0]
            )  # Cost in computing query hash value + locating the lower/upper bound in the hash table
            new_candidates = np.setdiff1d(
                np.where(collision_counts >= self.collision_threshold)[0], candidates
            )
            candidates = np.concatenate((candidates, new_candidates))
            candidates_dist = np.concatenate(
                (
                    candidates_dist,
                    np.linalg.norm(self.atoms[new_candidates] - query, axis=1),
                )
            )
            sample_complexity += (
                len(new_candidates) * self.atoms.shape[1] + self.atoms.shape[0]
            )  # Cost in getting new candidates + computing the distance of candidates
            if (len(candidates) != 0) and (
                np.sum(candidates_dist <= self.approx_const * radius) >= top_k
            ):
                return (
                    candidates[np.argsort(candidates_dist)[:top_k]],
                    int(sample_complexity),
                )
            else:
                collision_counts = np.zeros(
                    self.atoms.shape[0]
                )  # Reset collision counts
            # Radius update
            k = max(
                ceil(
                    log(
                        2 * np.median(nearest_atoms_dist) / self.width,
                        self.approx_const,
                    )
                ),
                k + 1,
            )
            radius = self.approx_const ** k
        return candidates[np.argsort(candidates_dist)[:top_k]], int(sample_complexity)


def toy_experiment_qalsh(seed: int = 0) -> None:
    """
     A toy experiment on QALSH with tabular data whose datapoints are drawn from a normal distribution.

    :param seed: Random seed number
    """
    np.random.seed(seed)
    atoms = np.random.normal(size=(300000, 10))
    query = np.random.normal(size=10)
    top_k = 5
    naive_ans = np.argsort(np.linalg.norm(atoms - query, axis=1))[:top_k]

    qalsh = QALSH(atoms=atoms, delta=0.1, approx_const=2.0)
    qalsh.generate_hash_tables()
    nn_search = qalsh.nn_search(query, top_k)
    sample_complexity_ratio = nn_search[1] / (atoms.shape[0] * atoms.shape[1])

    print(
        f"It is {naive_ans==nn_search[0]} that naive NN search accords with QALSH NN search. "
        f"The sample complexity of QALSH is {sample_complexity_ratio * 100}% of one of naive NN search"
    )


if __name__ == "__main__":
    toy_experiment_qalsh(1)
