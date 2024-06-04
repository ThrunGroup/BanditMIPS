from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
from surprise import Dataset
import numpy as np
import os

def main1():
    data = Dataset.load_builtin('ml-1m')
    print(data)
    trainset = data.build_full_trainset()

    model = NMF(verbose=True, n_epochs=30, biased=True)
    model.fit(trainset=trainset)

    cur_dir = os.path.dirname(__file__)
    np.save(os.path.join(cur_dir, "movie_lens_1m_users_factors.npy"), model.pu)
    np.save(os.path.join(cur_dir, "movie_lens_1m_users_biases.npy"), model.bu)
    np.save(os.path.join(cur_dir, "movie_lens_1m_items_factors.npy"), model.qi)
    np.save(os.path.join(cur_dir, "movie_lens_1m_items_biases.npy"), model.bi)
    np.save(os.path.join(cur_dir, "global_mean.npy"), trainset.global_mean)


def main2(filename: str = "movie_lens_1m_ratings"):
    """
    :param filename: Name of the file to store as a dataset
    """
    cur_dir = os.path.dirname(__file__)
    users_factors = np.load(os.path.join(cur_dir, "movie_lens_1m_users_factors.npy"))
    users_biases = np.load(os.path.join(cur_dir, "movie_lens_1m_users_biases.npy"))
    item_factors = np.load(os.path.join(cur_dir, "movie_lens_1m_items_factors.npy"))
    item_biases = np.load(os.path.join(cur_dir, "movie_lens_1m_items_biases.npy"))
    global_mean = np.load(os.path.join(cur_dir, "global_mean.npy"))

    data = item_factors @ (users_factors.transpose())
    data += np.expand_dims(item_biases, axis=1)
    data += + np.expand_dims(users_biases, axis=0)
    data += global_mean
    np.save(os.path.join(cur_dir, f"{filename}.npy"), data)
    print(np.histogram(data))
    print(data.shape)


if __name__ == "__main__":
    main1()
    main2()
