import pandas as pd
import numpy as np
import os
import tqdm
import glob


def main(
        with_volume: bool = False,
        filename: str = "crypto_pairs_1m_dimensions"
):
    """
    Make a dataset for 105 crypto pairs with 1M dimensions.

    :param with_volume: Whether to include "Volume" column in the dataset.
                        The column has exceptionally large values compared to others.
    :param filename: Name of the file to store as a dataset
    """
    directory_path = os.path.dirname(os.path.abspath(__file__))
    dataset_paths = glob.glob(os.path.join(directory_path, "*.npy"))
    if dataset_paths:
        print("Dataset already exists: ", dataset_paths)
        return

    dataset_directory = os.path.join(directory_path, "archive")
    num_atoms = 105  # Only about 100+ files have dimensions larger than 1M.
    if with_volume:
        # With a "volume" column, we can extract 5 features from each timestamp.
        # This tells us that we need to find files that have at least 200K timestamps,
        # and they are usually larger than 16MB.
        min_file_size = 16 * 1024 * 1024
    else:
        # Without a "volume" column, we can only extract 4 features from each timestamp.
        # Files with 250K timestamps are larger than 22MB.
        min_file_size = 19 * 1024 * 1024

    columns = ["open", "close", "high", "low"]
    if with_volume:
        columns += "volume",

    num_max_dimensions = 10 ** 6
    num_max_timestamps = num_max_dimensions // len(columns)

    files = []

    for file in os.listdir(dataset_directory):
        file_path = os.path.join(dataset_directory, file)

        if os.path.getsize(file_path) > min_file_size:
            files.append(file_path)

    files.sort(key=lambda file: os.path.getsize(file))
    files = files[:num_atoms]  # Only keep the crypto pairs that have more than 1M dimensions
    dataset = np.empty((len(files), num_max_dimensions))

    print("Making a crypto pairs dataset...")
    for (i, file) in tqdm.tqdm(enumerate(files), total=len(files)):
        # Flatten the first few rows of data to obtain a 1D atom
        df = pd.read_csv(file)
        dataset[i] = df[columns][:num_max_timestamps].values.flatten()

    np.save(os.path.join(directory_path, f"{filename}.npy"), dataset)

def preprocess_crypto_pairs():
    main(with_volume=False)


if __name__ == "__main__":
    main(with_volume=False)
