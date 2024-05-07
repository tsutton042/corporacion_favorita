"""
Get predictions from baseline models + optimised models
"""

import pandas as pd
from metrics import NWRMSLE, MAE, RMSE
from typing import Callable
from optimisation.opt import simulated_annealing
from modelling.baseline_models import persistance, moving_average
import json

# read in data
print("Reading data")
train = pd.read_csv("data/train_merged.csv")
test = pd.read_csv("data/test_merged.csv")

# massage NWRMLSLE using the weights of the current data
def get_data_NWRMSLE(weights: dict) -> Callable[[pd.Series, pd.Series], float]:
    def metric(preds: pd.Series, data: pd.Series) -> float:
        return NWRMSLE(preds, data, weights)

    return metric


def get_weights(data: pd.DataFrame) -> pd.Series:
    return pd.Series(1.25 if item else 1 for item in data.perishable)


train_nwrmsle = get_data_NWRMSLE(get_weights(train))
# test_nwrmsle = get_data_NWRMSLE(get_weights(test))

# make basleline predictions - on whole train set
# n4eed to do some grouping by store/item id! this prediction is technically not valid!
print("Running unoptimised persistance model")
pers_preds = persistance(train.unit_sales)
pers_results = {
    "nwrmsle": train_nwrmsle(pers_preds, train.unit_sales),
    "mae": MAE(pers_preds, train.unit_sales),
    "mse": RMSE(pers_preds, train.unit_sales),
}
results_proc = [
    ": ".join((k.upper(), str(v.round(5)))) for k, v in pers_results.items()
]
print("=" * 80)
print(f"Persistance model results: {results_proc}")
print("=" * 80)

print("Running unoptimised moving average model")
mv_av_preds = moving_average(train.unit_sales)
mv_av_results = {
    "nwrmsle": train_nwrmsle(mv_av_preds, train.unit_sales),
    "mae": MAE(mv_av_preds, train.unit_sales),
    "mse": RMSE(mv_av_preds, train.unit_sales),
}
results_proc = [
    ": ".join((k.upper(), str(v.round(5)))) for k, v in mv_av_results.items()
]
print("=" * 80)
print(f"Persistance model results: {results_proc}")
print("=" * 80)

# write results
with open("results/persist.json", "w") as f:
    json.dump(pers_results, f)
with open("results/mv_av.json", "w") as f:
    json.dump(mv_av_results, f)

# perform simulated annealing on the persistance model
# do not use the nwrmsle as the objective function to minimise
# as this is our evaluation metric! should avoid some level of overfitting
# Create a train & val set out of train for this purpose!
