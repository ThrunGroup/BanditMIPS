from exps.plot_baselines import create_scaling_plots
from utils.constants import SCALING_BASELINES_DATATYPES, SCALING_BASELINES_ALGORITHMS


def scaling_baselines_plot(
        algorithms,
        data_types,
        dir_name,
        save_to,
    ):
    """
    Function to generate the scaling plots of the mips algorithms that were run with scaling_baselines.
    The output is one plot with all of the baselines for each dataset.
    """
    create_scaling_plots(
        is_fit=False,
        include_error_bar=True,
        data_types=data_types,
        alg_names=algorithms,
        dir_name=dir_name,
        is_logspace_y=True,
        save_to=save_to,
    )


if __name__ == "__main__":
    scaling_baselines_plot(SCALING_BASELINES_ALGORITHMS)