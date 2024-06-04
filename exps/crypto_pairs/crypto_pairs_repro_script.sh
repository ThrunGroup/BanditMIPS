#!/bin/bash

# This script should be run inside the root directory of this project

pip install -e .
python data/crypto_pairs/preprocess_crypto_pairs.py  # Make a crypto pairs dataset
python exps/crypto_pairs/run_crypto_pairs_scaling.py  # Run the scaling experiment