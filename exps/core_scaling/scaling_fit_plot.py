from exps.plot_baselines import create_scaling_plots
from utils.constants import SCALING_FIT_DATATYPES, ACTION_ELIMINATION
import os

def scaling_fit_plot(
        algorithm,
        data_types,
        dir_name,
        save_to, 
):
    """
    Function to generate the fitted (linear, sqrt, log) scaling plots. 
    """
    create_scaling_plots(
        is_fit=True,
        include_error_bar=True,
        data_types=data_types,
        alg_names=[algorithm],
        dir_name=dir_name,
        is_logspace_x=False,
        is_logspace_y=False,
        save_to=save_to,
    )


if __name__ == "__main__":
    scaling_fit_plot([ACTION_ELIMINATION])#, ADAPTIVE_ACTION_ELIMINATION])