"""
Get predictions from baseline models + optimised models
"""

import pandas as pd
from metrics import NWRMSLE, MAE, RMSE
from typing import Callable
from modelling.baseline_models import persistance, moving_average
import json
from tqdm import tqdm

# read in data
print("Reading data")
train = pd.read_csv("data/train_merged.csv")
print("data read")
# massage NWRMLSLE using the weights of the current data
def get_data_NWRMSLE(weights: dict) -> Callable[[pd.Series, pd.Series], float]:
    def metric(preds: pd.Series, data: pd.Series) -> float:
        return NWRMSLE(preds, data, weights)

    return metric


def get_weights(perishable: pd.Series) -> pd.Series:
    return pd.Series(1.25 if item else 1 for item in perishable)


train_nwrmsle = get_data_NWRMSLE(get_weights(train.perishable))
# make basleline predictions - on whole train set
# n4eed to do some grouping by store/item id! this prediction is technically not valid!
print("Running unoptimised persistance model")
horizon = 1
pers_preds = []
ground_truth = []
perishable = []
# train_grouped = train.set_index(keys=["store_nbr", "item_nbr"])
store_item_map = pd.unique(pd.Series(zip(train.store_nbr, train.item_nbr)))
for store_num, item_num in tqdm(store_item_map):
    values = train[(train.store_nbr == store_num) & (train.item_nbr == item_num)]
    sales = values.unit_sales
    perish = values.perishable
    if horizon < values.shape[0]:
        preds = persistance(sales, horizon=horizon)
        pers_preds.extend(preds)
        ground_truth.extend(sales)
        perishable.extend(perish)
print(len(pers_preds))
# pers_preds = persistance(train.unit_sales)
pers_preds = pd.Series(pers_preds)
ground_truth = pd.Series(ground_truth)
perishable = pd.Series(perishable)
pers_nwrmsle = get_data_NWRMSLE(get_weights(perishable))
pers_results = {
    "nwrmsle": pers_nwrmsle(pers_preds, ground_truth),
    "mae": MAE(pers_preds, ground_truth),
    "mse": RMSE(pers_preds, ground_truth),
}
results_proc = [
    ": ".join((k.upper(), str(v.round(5)))) for k, v in pers_results.items()
]
print("=" * 80)
print(f"Persistance model results: {results_proc}")
print("=" * 80)

print("Running unoptimised moving average model")
window = 2
mv_av_preds = []
ground_truth = []
perishable = []
for store_num, item_num in tqdm(store_item_map):
    values = train[(train.store_nbr == store_num) & (train.item_nbr == item_num)]
    sales = values.unit_sales
    perish = values.perishable
    if window <= values.shape[0]:
        preds = moving_average(sales, window=window)
        mv_av_preds.extend(preds)
        ground_truth.extend(sales)
        perishable.extend(perish)
mv_av_preds = pd.Series(mv_av_preds)
ground_truth = pd.Series(ground_truth)
perishable = pd.Series(perishable)
mv_av_nwrmsle = get_data_NWRMSLE(get_weights(perishable))
mv_av_results = {
    "nwrmsle": mv_av_nwrmsle(mv_av_preds, ground_truth),
    "mae": MAE(mv_av_preds, ground_truth),
    "mse": RMSE(mv_av_preds, ground_truth),
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
