import os

from exps.core_scaling.scaling_fit_plot import scaling_fit_plot
from utils.constants import ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, NETFLIX


def scaling_noise_plot():
    """
    Function to generate the scaling plots of the MIPS algorithms that were run with scaling_baselines.
    The output is one plot with all of the baselines for each dataset.
    """
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.join(parent_dir, "logs")
    scaling_fit_plot(
        [ACTION_ELIMINATION],
        [NETFLIX],
        dir_name,
    )


if __name__ == "__main__":
    scaling_noise_plot()