import os

from exps.plot_baselines import create_tradeoff_plots
from utils.constants import (
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    NETFLIX,
)


def tradeoff_noise_plot():
    algorithms = [ACTION_ELIMINATION]
    data_types = [NETFLIX]
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "normalized_logs", f"topk_{top_k}")
        create_tradeoff_plots(
            alg_names=algorithms,
            data_types=data_types,
            top_k=top_k,
            log_dir=log_dir,
            max_speedup=None,
            is_logspace=True,
        )


if __name__ == "__main__":
    tradeoff_noise_plot()