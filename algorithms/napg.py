import numpy as np
import hnswlib
from typing import Tuple


def hnsw(
        atoms: np.ndarray,
        signals: np.ndarray,
        num_best_atoms: int = 1,
        # ef_construction: int = 8,
        # ef_search: int = 4,
        # num_links_per_node: int = 8,
        ef_construction: int = 1,
        ef_search: int = 1,
        num_links_per_node: int = 1,
        use_norm_adjusted_factors: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A hierarchical navigable small world algorithm that approximately solves MIPS using a graph.
    Implementation is mostly based on https://github.com/nmslib/hnswlib

    :param atoms: 2d array that contains atoms in rows. Atoms are the vectors one of which will be selected
                    as the vector with maximum inner product with a signal vector.
    :param signals: A matrix whose each row contains a signal vector.
    :param num_best_atoms: Number of atoms that bandit algorithm identify as top/best arms.
    :param ef_search: Number of the nearest neighbors to search for each node when searching the query.
                                The larger it is, the longer it takes to search the query, but the accuracy can
                                be higher.
    :param ef_construction: Number of the nearest neighbors to search for each node when constructing the graph.
                                The larger it is, the longer it takes to construct the graph, but the accuracy can
                                be higher.
    :param num_links_per_node: Number of bi-directional links between nodes
    :param use_norm_adjusted_factors: Whether to use NAPG proposed in "Norm Adjusted Proximity Graph for Fast
                                        Inner Product Retrieval." NAPG calculates the factors based on the norm
                                        of data to increase the chance of connecting more appropriate nodes.
    :return: Indices of best atom that have maximum inner product with a signal vector and the budget (total number of
            calculations)
    """
    num_atoms, num_dimensions = atoms.shape

    # Initialise a graph
    index = hnswlib.Index(space="ip", dim=num_dimensions)
    index.init_index(max_elements=num_atoms, ef_construction=ef_construction, M=num_links_per_node)

    # Data insertion
    ids = np.arange(num_atoms)
    index.add_items(atoms, ids)
    index.set_ef(ef_search)

    # Query
    labels, budget = index.knn_query(signals, k=num_best_atoms)
    labels.sort()

    budgets = np.array([budget]) / len(signals)

    return labels, budgets
