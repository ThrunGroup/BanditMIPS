from exps.plot_baselines import create_scaling_plots
from utils.constants import SCALING_FIT_DATATYPES, ACTION_ELIMINATION
import os

def scaling_fit_plot(
        algorithm,
        data_types=SCALING_FIT_DATATYPES,
        dir_name=None
):
    """
    Function to generate the fitted (linear, sqrt, log) scaling plots. Will output each baseline for each dataset.
    """
    create_scaling_plots(
        is_fit=True,
        include_error_bar=True,
        data_types=data_types,
        alg_names=[algorithm],
        dir_name="sample_complexity/",
        is_logspace_x=False,
        is_logspace_y=False,
    )


if __name__ == "__main__":
    scaling_fit_plot([ACTION_ELIMINATION])#, ADAPTIVE_ACTION_ELIMINATION])