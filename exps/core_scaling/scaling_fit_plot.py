from exps.plot_baselines import create_scaling_plots
from utils.constants import SCALING_BASELINES_DATATYPES, SCALING_BASELINES_ALGORITHMS, ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION
import os

def scaling_fit_plot(
        algorithms,
        data_types=SCALING_BASELINES_DATATYPES,
        dir_name=None
):
    """
    Function to generate the fitted (linear, sqrt, log) scaling plots. Will output each baseline for each dataset.
    """

    if dir_name is None:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "logs")
    else:
        log_dir = dir_name

    for algorithm in algorithms:
        create_scaling_plots(
            is_fit=True,
            include_error_bar=True,
            data_types=data_types,
            alg_names=[algorithm],
            dir_name=log_dir,
            is_logspace_x=False,
        )


if __name__ == "__main__":
    scaling_fit_plot([ACTION_ELIMINATION])#, ADAPTIVE_ACTION_ELIMINATION])