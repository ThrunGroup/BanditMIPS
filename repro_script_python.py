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
from utils.constants import SCALING_BASELINES_ALGORITHMS, ACTION_ELIMINATION, NETFLIX


def find_absent_algorithms(path, algorithms):
    absent = []
    for algorithm in algorithms:
        algo_logs = glob.glob(os.path.join(path, f"*{algorithm}*"))
        if len(algo_logs) == 0:
            absent.append(algorithm)

    return absent


def main(experiment):
    path = os.path.join(os.getcwd(), "exps")
    # idx = tranlate_experiment(experiment)

    if experiment == "noise":
        print("experiments for noise robustness")
        scaling_path = os.path.join(path, "eps_noise", "logs")
        scaling_algos = find_absent_algorithms(scaling_path, [ACTION_ELIMINATION])
        scaling_noise(scaling_algos)
        scaling_noise_plot()

        tradeoff_path = os.path.join(path, "eps_noise", "normalized_logs", "topk*")
        tradeoff_algos = find_absent_algorithms(tradeoff_path, [ACTION_ELIMINATION])
        print(tradeoff_path)
        tradeoff_noise(tradeoff_algos)
        tradeoff_noise_plot()
    else:
        scaling_path = os.path.join(path, "core_scaling", "logs")
        tradeoff_path = os.path.join(path, "core_tradeoff", "logs")

        # generate scaling/tradeoff log files
        scaling_baselines_algos = find_absent_algorithms(
            scaling_path, SCALING_BASELINES_ALGORITHMS
        )

        # Todo @jey, @luke: use this function accordingly... this is just an example.
        tradeoff_baselines_algos = find_absent_algorithms(
            tradeoff_path, SCALING_BASELINES_ALGORITHMS
        )

        # Todo @jey, @luke: need to add each of your scaling experiments here!
        print(f"=> Creating Scaling log files for {scaling_baselines_algos}")
        scaling_baselines(scaling_baselines_algos)

        # Todo @jey, @luke: need to add each of your tradeoff experiments here (included example below)!
        print(f"=> Creating Precision-Speed log files for {tradeoff_baselines_algos}")
        tradeoff_baselines()

        # generate the relative plots
        # Todo @jey, @luke: add scaling plot code here
        print("=> Generating scaling plots")
        scaling_baselines_plot(SCALING_BASELINES_ALGORITHMS)
        scaling_fit_plot(SCALING_BASELINES_ALGORITHMS)
        scaling_bucket_ae_plot()

        # Todo @jey, @luke: add tradeoff plot code here
        print("=> Generating tradeoff plots")
        tradeoff_baselines_plot()

        print("==> O(1) scaling on Highly Symmetric dataset")
        eps_suboptimal_exp()
        plot_eps_suboptimal()

        print("==> Sample Complexities scaling plots on OPT dataset")
        scaling_exp_gpt2()
        plot_scaling_gpt2(is_plot_runtime=False)

        print("==> Runtime Scaling plots on OPT, Movie Lens, and Netflix datasets")
        exp_runtime_scaling()


if __name__ == "__main__":
    main(sys.argv[1])
