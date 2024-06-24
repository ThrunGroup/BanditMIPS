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
    TRADEOFF_BASELINES_DATATYPES,
    ACTION_ELIMINATION,
    SCALING_FIT_DATATYPES,
    SCALING_BASELINES_DATATYPES,
    HIGHLY_SYMMETRIC,
)


def main(experiment):
    path = os.path.join(os.getcwd(), "exps")

    if experiment=="main":
        # figure 1: sample complexity
        complexity_path = os.path.join(path, "core_scaling", "sample_complexity")
        scaling_baselines([ACTION_ELIMINATION], SCALING_FIT_DATATYPES, complexity_path)
        scaling_fit_plot(
            algorithm=ACTION_ELIMINATION,
            data_types=SCALING_FIT_DATATYPES,
            dir_name=complexity_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure1:sample_complexities")
        )

        # figure 2: scaling comparisons
        scaling_path = os.path.join(path, "core_scaling", "scaling_comparison")
        scaling_baselines(SCALING_BASELINES_ALGORITHMS, SCALING_BASELINES_DATATYPES, scaling_path)
        scaling_baselines_plot(
            algorithms=SCALING_BASELINES_ALGORITHMS,
            data_types=SCALING_BASELINES_DATATYPES,
            dir_name=scaling_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure2:scaling_comparisons")
        )

        # figure 3: tradeoff @ precision 1
        tradeoff_path = os.path.join(path, "core_tradeoff", "normalized_logs")
        tradeoff_baselines(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[1],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
        )
        tradeoff_baselines_plot(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[1],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure3:tradeoff_k1")
        )

        # figure 4: high-dimensional datasets
        run_crypto_pairs_scaling(
            run=True, 
            plot=True, 
            save_to=os.path.join(os.getcwd(), "figures", "figure4:high-dimensional")
        )
        sift_scaling(
            run=True, 
            plot=True,
            save_to=os.path.join(os.getcwd(), "figures", "figure4:high-dimensional")
        )
    
    elif experiment=="appendix":
        # figure 5: tradeoff @ precision 5
        tradeoff_path = os.path.join(path, "core_tradeoff", "normalized_logs")
        tradeoff_baselines(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[5],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
        )
        tradeoff_baselines_plot(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[5],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure5:appendix_tradeoff_k5")
        )

        # figure 6: tradeoff @ precision 10
        tradeoff_path = os.path.join(path, "core_tradeoff", "normalized_logs")
        tradeoff_baselines(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[10],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
        )
        tradeoff_baselines_plot(
            algorithms=TRADEOFF_BASELINES_ALGORITHMS,
            top_ks=[10],
            data_types=TRADEOFF_BASELINES_DATATYPES,
            dir_name=tradeoff_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure6:appendix_tradeoff_k10")
        )

        # figure 7: compatibility with preprocessing + scaling with N
        scaling_bucket_ae()
        scaling_bucket_ae_plot()

        # figure 8: simple song fit
        song_scaling(
            run=True, 
            plot=True, 
            save_to=os.path.join(os.getcwd(), "figures", "figure8:appendix_simple_song")
        )

        # figure 9: symmetric normal
        complexity_path = os.path.join(path, "core_scaling", "sample_complexity")
        scaling_baselines([ACTION_ELIMINATION], [HIGHLY_SYMMETRIC], complexity_path)
        scaling_fit_plot(
            algorithm=ACTION_ELIMINATION,
            data_types=[HIGHLY_SYMMETRIC],
            dir_name=complexity_path,
            save_to=os.path.join(os.getcwd(), "figures", "figure9:appendix_highly-symmetric")
        )
    else:
        raise NameError("arg should be main or appendix")
    print(f"=> {experiment} figures generated!")



if __name__ == "__main__":
    main(sys.argv[1])
