import numpy as np
import math
import numba as nb

from typing import Iterable


@nb.njit
def binary_to_num(bnum: Iterable):
    """
    Converts an array of ints into one integer.
    [1, 0, 1, 1] => 1011
    """
    num = 0
    for idx in range(len(bnum)):
        num += bnum[-1 - idx] * (10 ** idx)
    return num


def g_index_extend(data, m):
    """
    [x] => [x;    ||x||**2; ||x||**4; ...; ||x||**(2*m)]
    """
    norms = np.linalg.norm(data, axis=1)
    columns = np.transpose([norms ** (2 * i) for i in range(1, m+1)])
    return np.concatenate([data, columns], axis=1)



def g_query_extend(query, m):
    """
    [x] => [x;    1/2; ...; 1/2]
    """
    return query + [0.5 for i in range(m)]


def g_index_simple_extend(data):
    """
    This should only be called when we're using symmetric hash functions.
    [x] => [x;   sqrt(1 - ||x||**2)]
    """
    return np.concatenate([data, (data * data).sum(axis=1, keepdims=True)], axis=1)


def g_query_simple_extend(query):
    """
    [x] => [x;   sqrt(1 - ||x||**2)]
    """
    assert round(np.dot(query, query)) == 1
    return query + [0]


def g_transformation(data):
    """
    A scaling transformation S(.) to convert MIPS to NN problem
    Conversion is as follows (U < 1  ||xi||_2 <= U <= 1):
        S(x) = x * U/M
    where M = max_i ||x_i||_2
    """
    U = 0.83
    norms = np.linalg.norm(data, axis=1)
    max_norm = np.max(norms)
    ratio = U / max_norm
    return ratio, max_norm, ratio * data


def g_normalization(query):
    """
    Normalize the query vector so that ||q||_2 = 1
    """
    if int(np.sum(query == 0)):
        len_ = len(query)
        return [q / len_ for q in query]

    norm = math.sqrt(np.dot(query, query))
    return [q / norm for q in query]