"""
Defines some basic optimization algorithms
"""

import pandas as pd
from typing import Callable
import numpy as np


def simulated_annealing(
    func: Callable[..., pd.Series],
    metric: Callable[[pd.Series, pd.Series], float],
    data: pd.Series,
    initial_params: dict,
    neighbour_func: Callable[[dict], dict],
    n_iter: int = 100,  # algorithms this is planned to be used with are not expensive to run
    init_temp: float = 60,  # some arbitrary but reasonable-feeling value
    cooling_rate: float = 0.9,  # seems reasonably slow
    allow_diff_lengths: bool = False,
    seed: int = 20624,  # randomly generated seed for reproduceability
) -> tuple[dict, pd.Series, float]:
    """
    Runs a basic simulated annealing (minimisation) algorithm, using an exponential cooling schedule.
    Takes into account the fact we may be optimising a horizon/window
    """
    rng = np.random.default_rng(seed)
    params = initial_params
    temp = init_temp / cooling_rate  # undo first iter of cooling schedule
    preds = func(data, **params)
    diff_lengths = (data.shape[0] - preds.shape[0]) if allow_diff_lengths else 0
    curr_score = metric(preds, data.iloc[diff_lengths:])
    for _ in range(n_iter):
        temp = cooling_rate * temp  # exponential cooling schedule
        new_params = neighbour_func(params)
        new_preds = func(data, **params)
        diff_lengths = (data.shape[0] - new_preds.shape[0]) if allow_diff_lengths else 0
        new_score = metric(new_preds, data.iloc[diff_lengths:])
        if (new_score < curr_score) or (
            np.exp((new_score - curr_score) / temp) < rng.uniform()
        ):  # Metropolis-Hastings criterion as is a common choice of acceptance criteria
            params = new_params
            preds = new_preds
            curr_score = new_score
    return params, preds, curr_score
