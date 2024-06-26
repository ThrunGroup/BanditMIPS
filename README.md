# Faster Maximum Inner Product Search in High Dimensions

# BanditMIPS algorithm:
BanditMIPS is a novel randomized algorithm for the Maximum Inner Product Search (MIPS) problem, commonly encountered in machine learning applications such as recommendation systems. This algorithm addresses the challenge of high-dimensional vector spaces by utilizing a complexity that is independent of dimension size. It does so by employing an adaptive sampling strategy inspired by multi-armed bandits, providing PAC guarantees while significantly reducing computational overhead. Some other notable featuers are as follows:
- No preprocessing required (although common preprocessing techniques can be used to further accelerate the algorithm)
- Configurable hyperpameter for accuracy-speed tradeoff
- BanditMIPS-$\alpha$ variant that offers faster results through non-uniform sampling across dimensions.

# Installation
All experiments were run with `python=3.10`. Please run `pip install -r requirements.txt` to install all dependencies. 

# Obtaining / Preprocessing the Datasets
- The Movie Lens 1m data is available from [here](https://grouplens.org/datasets/movielens/1m/).
  1. Run `python -m data.movie_lens.preprocess_movie_lens_1m`
- The Netflix Prize dataset is available from [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
  1. Download the `combined_data_1.txt` file and move to `data/netflix/combined_data_1.txt`.
  2. Run `python -m data.netflix.preprocess_netflix`
- The Crypto Pairs (1M dimensions) dataset is available from [here](https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution).
  1. Download the "archive" zip file and extract it inside `data/crypto_pairs`
  2. Run `python -m data.crypto_pairs.preprocess_crypto_pairs`

# Reproducing the results
Please run `chmod +x repro_script.sh && ./repro_script.sh <main / appendix>` to reproduce all of the experimental logs in the main / appendix of the paper. Each of the figures will be generated under the `figures` folder with their corresponding figure names.

# Description of Files

The files are organized as follows:

- `algorithms/` contains the different MIPS algorithms: BanditMIPS is implemented in `action_elimination.py` and BanditMIPS-$\alpha$ is implemented in `adaptive_action_elimination.py`. The most recently-developed baseline algorithm, BoundedME, is implemented in `median_elimination.py`.
- `data/` contains the data used in the experiments. If you choose to reproduce the results, please download the data files into their respective folders and run `data/netflix/preproces_netflix.py` and `data/movie_lens/preprocess_movie_lens_1m.py`.
- `exps/` contains the scripts for the experiments.
  - the two main types of experiments are the scaling experiments and the precision/speedup tradeoff experiments
- `utils/` contains helper functions used across the codebase.
- `logs/` contains the logs of the experiments.
- `figures/` contains the figures from analyzing the experimental results.

# Credits

The implementation of Norm Adjusted Proximity Graph for Fast Inner Product Retrieval uses code from the hnswlib repository (https://github.com/nmslib/hnswlib) under the Apache 
2.0 license. Thank you to the authors for their contributions. It is licensed under the Apache 2.0 license. A copy 
of the license can be found at http://www.apache.org/licenses/LICENSE-2.0.
