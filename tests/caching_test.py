import numpy as np
import numba as nb
from utils.utils import subset_2d_cached


def print_datastructures(subset_array, cache, cache_tracker, cache_map):
    print(f"subset_array: \n{subset_array}\n")
    print(f"cache: \n{cache}\n")
    print(f"tracker: \n{cache_tracker}\n")
    print(f"cache_map (cols -> cache_idx): \n{cache_map}\n")


def validate_input(str_list, rows_list, cols_list):
    assert len(str_list) == len(rows_list), "length of str_list and rows_list must be equivalent"
    assert len(str_list) == len(cols_list), "length of rows_list and cols_list must be equivalent"

    for i in range(len(str_list)):
        actions = str_list[i].split(",")
        num_elems = 0
        for action in actions:
            num_elems += int(action.split()[1])
        assert num_elems == (len(rows_list[i]) * len(cols_list[i])), \
            "number of elements is not consistent with description"

    print("=> input description is valid")


def true_budget(action_seq):
    budget = 0.0
    actions = action_seq.split(",")
    for action in actions:
        a, num_elems = action.split()
        if a == "sample":
            budget += float(num_elems)
        elif a == "write":
            budget += float(num_elems) * 1.01
        else:
            budget += float(num_elems) * 0.01

    return budget


def main(verbose=False):
    source_array = np.arange(30).reshape((3, 10))
    cache = - np.ones((3, 4))
    cache_tracker = np.zeros(3, dtype=np.int64)  # each element < 4 since this is cache size
    cache_map = nb.typed.List()
    for i in range(3):
        cache_map.append(nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64))

    str_list = ["write 4", "read 4", "write 4", "read 2, sample 2", "write 1", "write 1, sample 1, sample 1"]
    rows_list = [[0, 2], [0, 2], [0, 2], [0, 2], [1], [0, 1, 2]]
    cols_list = [[0, 2], [0, 2], [3, 4], [4, 5], [6], [7]]
    validate_input(str_list, rows_list, cols_list)
    inputs = zip(rows_list, cols_list)

    if verbose:
        print(f"original array: \n {source_array}\n")
        print("----------------------------------")
        print_datastructures([], cache, cache_tracker, cache_map)

    for i, input in enumerate(inputs):
        subset_array, budget, cache, cache_tracker, cache_map, _ = subset_2d_cached(
            source_array, np.array(input[0]), np.array(input[1]), True, cache, cache_tracker, cache_map
        )
        assert true_budget(str_list[i]) - budget < 0.0001, "budget computation is incorrect"
        if verbose:
            print(f"-> {str_list[i]} with budget {budget}\n")
            print_datastructures(subset_array, cache, cache_tracker, cache_map)

    print("=> all caching tests passed")


if __name__ == "__main__":
    main(verbose=False)