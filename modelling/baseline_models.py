"""
Defines basline models for comparison purposes. These will not be put into the kaggle compettion page!
"""

import pandas as pd
from typing import Union


def persistance(data: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Defines a basic persistance model with a defined horizon
    Forms a baseline to compare other models to.
    Note that the model cheats for the first horizon predictions!
    """
    assert horizon > 0, "Horizon is not positive!"
    assert data.shape[0] > horizon, "Horizon is longer than data!"
    preds = list(data.iloc[:horizon])
    for i in range(horizon, data.shape[0]):
        preds.append(data.iat[i - horizon])
    return pd.Series(preds)


def moving_average(
    data: pd.Series,
    window: int = 2,
    weights: Union[list[float], None] = None,
    match_lengths: bool = True,
) -> pd.Series:
    """
    Defines a moving (weighted) average with a given window.
    If window is None, then performs an unweighted average instead.
    """
    # handle the case where weights is None
    if weights is None:
        weights = [1 for _ in range(window)]
    assert (
        len(weights) >= window
    ), f"Less weights ({len(weights)}) than values in the window ({window})"
    assert (
        len(weights) <= window
    ), f"More weights ({len(weights)}) than values in the window ({window})"
    assert all(x >= 0 for x in weights), "Not all weights are positive"
    # convert weights to a unit-sum window function - means no need to divide later
    unit_weights = pd.Series(x / sum(weights) for x in weights)
    # do this hack-job to make sure the inputs are as long as the outputs
    if match_lengths:
        preds = [data.iloc[0]]
        for i in range(1, window):
            preds.append(
                (data.iloc[:i] * unit_weights.iloc[:i]).sum()
                / unit_weights.iloc[:i].sum()
            )
    else:
        preds = []
    # now actually run the usual moving average
    for i in range(window, data.shape[0]):
        windowed_data = data.iloc[(i - window) : i]
        preds.append((windowed_data * unit_weights).sum())
    return pd.Series(preds)
