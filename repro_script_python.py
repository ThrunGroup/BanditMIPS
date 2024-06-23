import os
import glob
import sys

from exps.core_scaling.scaling_baselines import scaling_baselines
from exps.core_scaling.bucket_ae_scaling.bucket_action_elimination_scaling import (
    scaling_bucket_ae,
    scaling_bucket_ae_plot,
)
from exps.core_scaling.scaling_baselines_plot import scaling_baselines_plot
from exps.core_scaling.scaling_fit_plot import scaling_fit_plot
from exps.core_tradeoff.tradeoff_baselines import (
    tradeoff_baselines,
    tradeoff_baselines_plot,
)
from exps.plot_baselines import generate_scaling_plots, generate_tradeoff_plots
from exps.large_scaling.large_scaling_exps import large_scaling_exps
from exps.eps_noise.scaling_noise import scaling_noise
from exps.eps_noise.scaling_noise_plot import scaling_noise_plot
from exps.eps_noise.tradeoff_noise import tradeoff_noise
from exps.eps_noise.tradeoff_noise_plot import tradeoff_noise_plot
from exps.rebuttal_exps.runtime_scaling import exp_runtime_scaling
from exps.rebuttal_exps.eps_suboptimal import eps_suboptimal_exp, plot_eps_suboptimal
from exps.rebuttal_exps.scaling_exp_gpt2 import scaling_exp_gpt2, plot_scaling_gpt2
from exps.crypto_pairs.run_crypto_pairs_scaling import run_crypto_pairs_scaling
from exps.high_dimension.run_sift_scaling import sift_scaling
from exps.high_dimension.run_song_scaling import song_scaling
from utils.constants import SCALING_BASELINES_ALGORITHMS, TRADEOFF_BASELINES_ALGORITHMS, ACTION_ELIMINATION


def find_absent_algorithms(path, algorithms, tradeoff=False):
    absent = []
    if tradeoff:
        paths = [os.path.join(path, "topk_1"), os.path.join(path, "topk_10")]
    else:
        paths = [path]

    for path in paths:
        for algorithm in algorithms:
            algo_logs = glob.glob(os.path.join(path, f"*{algorithm}*"))
            if len(algo_logs) == 0:
                absent.append(algorithm)

    return absent


def main():
    path = os.path.join(os.getcwd(), "exps")

    # # scaling comparisons
    # scaling_path = os.path.join(path, "core_scaling", "logs")
    # scaling_baselines_algos = find_absent_algorithms(
    #     scaling_path, SCALING_BASELINES_ALGORITHMS
    # )
    # scaling_baselines(scaling_baselines_algos)
    # scaling_baselines_plot(SCALING_BASELINES_ALGORITHMS)

    # # tradeoff comparisons
    # tradeoff_path = os.path.join(path, "core_tradeoff", "normalized_logs")
    # tradeoff_baselines_algos = find_absent_algorithms(
    #     tradeoff_path, TRADEOFF_BASELINES_ALGORITHMS, tradeoff=True,
    # )
    # tradeoff_baselines(tradeoff_baselines_algos)
    # tradeoff_baselines_plot(TRADEOFF_BASELINES_ALGORITHMS)

    # # sample complexity
    # scaling_fit_plot(ACTION_ELIMINATION)
    # run_crypto_pairs_scaling(run=True, plot=True)
    # sift_scaling(run=True, plot=True)
    # song_scaling(run=True, plot=True)
    
    # # compatibility with preprocessing 
    # scaling_bucket_ae()
    # scaling_bucket_ae_plot()

    # runtime 
    print("==> Runtime Scaling plots on OPT, Movie Lens, and Netflix datasets")
    exp_runtime_scaling()


if __name__ == "__main__":
    main()
