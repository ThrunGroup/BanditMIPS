import os
import glob
import sys

from exps.core_scaling.scaling_baselines import scaling_baselines
from exps.core_scaling.bucket_ae_scaling.bucket_action_elimination_scaling import scaling_bucket_ae, scaling_bucket_ae_plot
from exps.core_scaling.scaling_baselines_plot import scaling_baselines_plot
from exps.core_scaling.scaling_fit_plot import scaling_fit_plot
from exps.core_tradeoff.tradeoff_baselines import tradeoff_baselines, tradeoff_baselines_plot
from exps.rebuttal_exps.runtime_scaling import exp_runtime_scaling
from exps.crypto_pairs.run_crypto_pairs_scaling import run_crypto_pairs_scaling
from exps.high_dimension.run_sift_scaling import sift_scaling
from exps.high_dimension.run_song_scaling import song_scaling
from utils.constants import (
    SCALING_BASELINES_ALGORITHMS, 
    TRADEOFF_BASELINES_ALGORITHMS, 
    ACTION_ELIMINATION,
    SCALING_FIT_DATATYPES,
    SCALING_BASELINES_DATATYPES,
)


def main(experiment):
    path = os.path.join(os.getcwd(), "exps", "core_scaling")

    if experiment=="main":
        # figure 1: sample complexity
        complexity_path = os.path.join(path, "sample_complexity")
        scaling_baselines([ACTION_ELIMINATION], SCALING_FIT_DATATYPES, complexity_path)
        scaling_fit_plot(
            algorithm=ACTION_ELIMINATION,
            data_types=SCALING_FIT_DATATYPES,
            dir_name=complexity_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure1:sample_complexities")
        )

        # figure 2: scaling comparisons
        scaling_path = os.path.join(path, "scaling_comparison")
        scaling_baselines(SCALING_BASELINES_ALGORITHMS, SCALING_BASELINES_DATATYPES, scaling_path)
        scaling_baselines_plot(
            algorithms=SCALING_BASELINES_ALGORITHMS,
            data_types=SCALING_BASELINES_DATATYPES,
            dir_name=scaling_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure2:scaling_comparisons")
        )

        import ipdb; ipdb.set_trace()
        # tradeoff comparisons
        tradeoff_path = os.path.join(path, "core_tradeoff", "normalized_logs")
        tradeoff_baselines(tradeoff_baselines_algos)
        tradeoff_baselines_plot(TRADEOFF_BASELINES_ALGORITHMS)

        # sample complexity
        scaling_fit_plot(ACTION_ELIMINATION)
        run_crypto_pairs_scaling(run=True, plot=True)
        sift_scaling(run=True, plot=True)
        song_scaling(run=True, plot=True)
        
        # compatibility with preprocessing 
        scaling_bucket_ae()
        scaling_bucket_ae_plot()

        # runtime 
        print("==> Runtime Scaling plots on OPT and Netflix datasets")
        exp_runtime_scaling()


if __name__ == "__main__":
    main(sys.argv[1])
