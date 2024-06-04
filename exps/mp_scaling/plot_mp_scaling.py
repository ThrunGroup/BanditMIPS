import os

from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    ACTION_ELIMINATION,
    SIMPLE_SONG,
)

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(parent_dir, "logs")

    create_scaling_plots(
        is_fit=True,
        is_logspace=True,
        include_error_bar=True,
        dir_name=log_dir,
        caching_type=os.path.join("mp_scaling", "logs"),
        data_types=[SIMPLE_SONG],
        alg_names=[ACTION_ELIMINATION]
    )
