import numpy as np
from typing import Tuple, List


class Tree:
    """
    Basic Tree class to build the pca-tree data structure

    :j: the index that is delta positions from the end.
    """

    def __init__(self):
        self.data = None
        self.root = None

    @staticmethod
    def first_reduction(
        arr: np.ndarray,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Apply the first reduction in the preprocessing step and return the concatenated arr.
        Conversion is described below where phi = max_i ||x_i||, and ";" signifies concatenation.
        (X is the dataset matrix and y is the query vector):
            x_i -> [x_i; sqrt(phi^2 - ||x_i||^2)]
            y -> [y; 0]

        :arr: 1d array if is_query, otherwise 2d array
        :returns: The concatenated array
        """
        if is_query:
            return np.append(arr, 0)

        norms = np.linalg.norm(arr, axis=1)
        phi = np.max(norms)
        temp = phi**2 - norms**2
        extra_dim = np.sqrt(
            np.where(temp < 0, 0, temp)
        )
        column = extra_dim.reshape(norms.shape[0], 1)
        return np.append(column, arr, axis=1)

    @staticmethod
    def second_reduction(
        arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the second reduction in the pre-processing step and return the transformed arr.
        Note that the components are in decreasing order of variance.
        Converts X -> W^T (X - mu) where
            - mu: 1/n * sum(X_i)
            - (X - M) = W Sig U^T where M is mu concatenated as columns

        :returns: the pca matrix, mu, and the transformed array
        """
        N, D = arr.shape
        mu = np.mean(arr, axis=0)

        M = np.repeat(mu.reshape((D, 1)), N, axis=1)
        assert M.shape == (D, N)

        W, _, _ = np.linalg.svd(arr.T - M, full_matrices=True)
        assert W.shape == (D, D)

        return W, mu, np.matmul(W.T, (arr - mu).T)

    def pca_tree_rec(
        self,
        node: object,
        j: int,
        max_j: int
    ) -> None:
        """
        Recursive helper function that helps build the pca-tree.

        :node: the current node
        :j: the dimension that we want to compare
        :max_j: dimension of the best principal component
        """
        if j == max_j or len(node.idcs) <= 1:
            return

        # find median value of node data
        node_col = self.data[node.idcs][:, j]
        m = np.median(node_col)
        node.m = m
        node.j = j

        # initialize children
        l_idcs = node.idcs[np.where(node_col <= m)]
        r_idcs = node.idcs[np.where(node_col > m)]
        node.left = Node(l_idcs)
        node.right = Node(r_idcs)

        self.pca_tree_rec(node.left, j + 1, max_j)
        self.pca_tree_rec(node.right, j + 1, max_j)
        return

    def construct_pca_tree(
        self,
        j: int
    ) -> object:
        """
        Construct the pca-tree (similar to kd-trees) data structure.

        :j: the index that is delta positions from the end where delta is the
            number of principle components that you want to use as the dimensions of the split.
        :returns: the root node
        """
        N, D = self.data.shape
        root = Node(np.arange(N))
        self.pca_tree_rec(root, j, D)
        self.root = root
        return root

    def print_tree_rec(
        self,
        node,
        space
    ) -> None:
        """
        Debugging function that prints the contents of the tree

        :node: the root node
        :space: indent space for printing
        """
        if node.m is None:
            print(space + "node idcs: ", node.idcs)
            return

        print(space + "node idcs: ", node.idcs)

        print(space + "--left: ")
        self.print_tree_rec(node.left, space + "    ")

        print(space + "--right: ")
        self.print_tree_rec(node.right, space + "    ")


class Node:
    def __init__(self, idcs):
        self.idcs = idcs
        self.m = None
        self.j = None

        self.left = None
        self.right = None
