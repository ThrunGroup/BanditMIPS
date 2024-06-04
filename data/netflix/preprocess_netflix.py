import pandas as pd
import numpy as np
import math
import re
import os
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
from surprise.model_selection import cross_validate


def main1():
    # Follow the codes from Kaggle, see
    is_file_exist = True
    if not is_file_exist:
        filename = os.path.dirname(os.path.dirname(__file__))
        for file in ["data", "netflix", "combined_data_1.txt", "combined_data_1.txt"]:
            filename = os.path.join(filename, file)
        df = pd.read_csv(filename, header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

        df['Rating'] = df['Rating'].astype(float)

        print('Dataset 1 shape: {}'.format(df.shape))
        print('-Dataset examples-')
        print(df.iloc[:100, :])

        df_nan = pd.DataFrame(pd.isnull(df.Rating))
        df_nan = df_nan[df_nan['Rating'] == True]
        df_nan = df_nan.reset_index()

        movie_np = []
        movie_id = 1

        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            # numpy approach
            temp = np.full((1, i - j - 1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1

        # Account for last record and corresponding length
        # numpy approach
        last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        print('Movie numpy: {}'.format(movie_np))
        print('Length: {}'.format(len(movie_np)))

        # remove those Movie ID rows
        df = df[pd.notnull(df['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)
        print('-Dataset examples-')
        print(df.iloc[::5000000, :])

        f = ['count', 'mean']

        df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
        df_movie_summary.index = df_movie_summary.index.map(int)
        movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
        drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

        print('Movie minimum times of review: {}'.format(movie_benchmark))

        df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
        df_cust_summary.index = df_cust_summary.index.map(int)
        cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
        drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

        print('Customer minimum times of review: {}'.format(cust_benchmark))

        print('Original Shape: {}'.format(df.shape))
        df = df[~df['Movie_Id'].isin(drop_movie_list)]
        df = df[~df['Cust_Id'].isin(drop_cust_list)]
        print('After Trim Shape: {}'.format(df.shape))
        print('-Data Examples-')
        print(df.iloc[::5000000, :])
        df.to_csv("netflix_cleaned1")

    # Fit SVD model and store arrays
    df = pd.read_csv("netflix_cleaned1")
    reader = Reader()
    data = Dataset.load_from_df(df[['Movie_Id', 'Cust_Id', 'Rating']][:], reader)
    trainset = data.build_full_trainset()
    model = SVD(verbose=True)

    print("start fitting")
    model.fit(trainset=trainset)

    cur_dir = os.path.dirname(__file__)
    np.save(os.path.join(cur_dir, "Movie_factors_15_new.npy"), model.pu)
    np.save(os.path.join(cur_dir, "Movie_biases_15_new.npy"), model.bu)
    np.save(os.path.join(cur_dir, "Customer_factors_15_new.npy"), model.qi)
    np.save(os.path.join(cur_dir, "Customer_biases_15_new.npy"), model.bi)
    np.save(os.path.join(cur_dir, "netflix_global_mean.npy"), trainset.global_mean)
    print("end fitting")


def main2(filename: str = "Movie_ratings"):
    cur_dir = os.path.dirname(__file__)
    movie_factors = np.load(os.path.join(cur_dir, "Movie_factors_15_new.npy"))
    customer_factors = np.load(os.path.join(cur_dir, "Customer_factors_15_new.npy"))
    movie_biases = np.load(os.path.join(cur_dir, "Movie_biases_15_new.npy"))
    customer_biases = np.load(os.path.join(cur_dir, "Customer_biases_15_new.npy"))
    global_mean = np.load(os.path.join(cur_dir, "netflix_global_mean.npy"))

    data = movie_factors @ (customer_factors.transpose())
    data += np.expand_dims(movie_biases, axis=1)
    data += + np.expand_dims(customer_biases, axis=0)
    data += global_mean

    filename = os.path.join(cur_dir, f"{filename}.npy")
    np.save(filename, data)
    print("Store Movie ratings matrix.")


if __name__ == "__main__":
    main1()
    main2()
