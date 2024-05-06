import pandas as pd
import numpy as np


def NWRMSLE(pred: pd.Series, actual: pd.Series, weights: pd.Series) -> float:
    # ensure input data are 1-d arrays
    assert (
        len(pred.shape) == 1
    ), f"Predictions are not a 1-d array, but have shape {pred.shape}"
    assert (
        len(actual.shape) == 1
    ), f"Ground-truth values are not a 1-d array, but have shape {actual.shape}"
    assert (
        len(weights.shape) == 1
    ), f"Weights are not a 1-d array, but have shape {weights.shape}"
    # verify the data are of the same length
    assert (
        pred.shape[0] == actual.shape[0]
    ), "Predictions and ground-truth are not the same size!"
    assert weights.shape[0] == pred.shape[0], "Weights and data not the same size!"
    # calculate metric - slight modification to allow for proper evaluation of returns
    numer = weights * (np.log(pred.abs() + 1) - np.log(actual.abs() + 1))
    return np.sqrt(np.power(numer, 2).sum() / weights.sum())


def MAE(pred: pd.Series, actual: pd.Series) -> float:
    # ensure input data are 1-d arrays
    assert len(pred.shape) == 1, "Predictions are not a 1-d array!"
    assert len(actual.shape) == 1, "Ground-truth values are not a 1-d array!"
    # verify the data are of the same length
    assert (
        pred.shape[0] == actual.shape[0]
    ), "Predictions and ground-truth are not the same size!"
    # calculate metric
    return np.absolute(pred - actual).sum() / pred.shape[0]


def RMSE(pred: pd.Series, actual: pd.Series) -> float:
    # ensure input data are 1-d arrays
    assert len(pred.shape) == 1, "Predictions are not a 1-d array!"
    assert len(actual.shape) == 1, "Ground-truth values are not a 1-d array!"
    # verify the data are of the same length
    assert (
        pred.shape[0] == actual.shape[0]
    ), "Predictions and ground-truth are not the same size!"
    # calculate metric
    return np.sqrt(np.power(pred - actual, 2).sum() / pred.shape[0])
