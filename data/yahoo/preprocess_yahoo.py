import numpy as np
import pandas as pd
import os

import math
import re
import os
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate


def main1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    yahoo_fp = os.path.join(dir_path, "../..", "data", "raw_data", "yahoo_dataset_array.npy")
    yahoo_numpy = np.load(yahoo_fp)

    # TODO: Need to add headers to yahoo_numpy to use pandas code?
    df = pd.read_csv(yahoo_fp)
    reader = Reader()
    data = Dataset.load_from_df(df[['Movie_Id', 'Cust_Id', 'Rating']][:], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=20)
    print("start fitting")
    model.fit(trainset=trainset)
    print(model.pu.shape)
    print(model.qi.shape)
    np.save("Yahoo_items_factors_20_new", model.pu)
    np.save("Yahoo_items_biases_20_new", model.bu)
    np.save("Yahoo_Customer_factors_20_new", model.qi)
    np.save("Yahoo_Customer_biases_20_new", model.bi)


def main2(filename: str = "Yahoo_ratings"):
    """
    Save the preprocessed Yahoo dataset as a file

    :param filename: Name of the file to store as a dataset
    """
    item_factors = np.load("Yahoo_items_factors_20_new.npy")
    item_biases = np.load("Yahoo_items_biases_20_new.npy")
    customer_factors = np.load("Yahoo_Customer_factors_20_new.npy")
    customer_biases = np.load("Yahoo_Customer_biases_20_new.npy")

    yahoo_data = (item_factors + np.expand_dims(item_biases, axis=1)) @ \
           ((customer_factors + np.expand_dims(customer_biases, axis=1)).transpose())
    np.save(filename, yahoo_data)


if __name__ == "__main__":
    main1()
    main2()