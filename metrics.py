import pandas as pd


def NWRMSLE(pred: pd.Series, actual: pd.Series, weights: pd.Series) -> float:
    # ensure input data are 1-d arrays
    assert len(pred.shape) == 1, "Predictions are not a 1-d array!"
    assert len(actual.shape) == 1, "Ground-truth values are not a 1-d array!"
    assert len(weights.shape) == 1, "Weights are not a 1-d array!"
    # verify the data are of the same length
    assert (
        pred.shape[0] == actual.shape[0]
    ), "Predictions and ground-truth are not the same size!"
    assert weights.shape[0] == pred.shape[0], "Weights and data not the same size!"
    # calculate metric
    numer = weights * (np.log(pred + 1) - np.log(actual + 1))
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


def MSE(pred: pd.Series, actual: pd.Series) -> float:
    # ensure input data are 1-d arrays
    assert len(pred.shape) == 1, "Predictions are not a 1-d array!"
    assert len(actual.shape) == 1, "Ground-truth values are not a 1-d array!"
    # verify the data are of the same length
    assert (
        pred.shape[0] == actual.shape[0]
    ), "Predictions and ground-truth are not the same size!"
    # calculate metric
    return np.pow(pred - actual, 2).sum() / pred.shape[0]
