"""
File to process data into a useable time-series format.
Output is saved to disk so that the processing only needs to be done once
"""

import numpy as np
import pandas as pd
from common import lin_interp, interp_oil
from pickle import dump


print("Reading data")
data = pd.read_csv("data/train_merged.csv")
# do linear interpolation on dcoilwtico
print("Interpolating oil prices")
oil_prices = data.groupby("date").first().dcoilwtico
interp_oil_prices = lin_interp(oil_prices)
data = interp_oil(data, interp_oil_prices)
# make indicator features - see EDA for justification of these
print("Creating indicator columns")
data["produce"] = data.family == "PRODUCE"
data["type_A"] = data.type == "A"
data["beverage"] = data.family == "BEVERAGES"
data["poultry"] = data.family == "POULTRY"
data["meats"] = data.family == "MEATS"
data["prepared_foods"] = data.family == "PREPARED FOODS"
data["school_office_supplies"] = data.family == "SCHOOL AND OFFICE SUPPLIES"
# make a train/val split on store/item split
print("Creating train/val sets")
store_item_map = pd.unique(pd.Series(zip(data.store_nbr, data.item_nbr)))
rng = np.random.default_rng(seed=890867)
# limit the size of these manually or we will have a *stupid* amount of training data
val_store_item = rng.choice(store_item_map, size=25)
rest_store_item = store_item_map[
    [x not in list(val_store_item) for x in store_item_map]
]
train_store_item = rest_store_item[:250]
test_store_item = rest_store_item[250:400]
print("This will take a while")
train = data[[x in list(train_store_item) for x in zip(data.store_nbr, data.item_nbr)]]
val = data[[x in list(val_store_item) for x in zip(data.store_nbr, data.item_nbr)]]
test = data[[x in list(test_store_item) for x in zip(data.store_nbr, data.item_nbr)]]
# structure as a time series - we form one for each time step of each store/item pair
print("Making time series structures")
feature_cols = [
    "dcoilwtico",
    "day_off",
    "produce",
    "type_A",
    "beverage",
    "poultry",
    "meats",
]

features_2 = [
    "onpromotion",
    "produce",
    "type_A",
    "beverage",
    "perishable",
    "poultry",
    "day_off",
    "meats",
    "prepared_foods",
    "school_office_supplies",
]


def structure_ts(tab_data, features, window_size=8):
    series = []
    labels = []
    store_item_map = pd.unique(pd.Series(zip(tab_data.store_nbr, tab_data.item_nbr)))
    for store, item in store_item_map:
        proc_pair = tab_data[
            (tab_data.store_nbr == store) & (tab_data.item_nbr == item)
        ]
        for date in proc_pair.date:
            proc_data = proc_pair[proc_pair.date <= date]
            subset = proc_data.loc[:, features]
            if window_size > proc_data.shape[0]:
                # front-pad it with columns of all 0 before the data
                subset = np.concatenate(
                    (
                        np.zeros((window_size - proc_data.shape[0], len(features))),
                        np.array(subset.astype(float)),
                    ),
                    axis=0,
                    dtype=float,
                )
            else:
                # take the last window_size options
                subset = np.array(subset.iloc[-window_size:, :].astype(float))
            series.append(subset)
            labels.append(proc_data[proc_data.date == date].unit_sales)
    return np.array(series), np.array(labels)


train_x, train_y = structure_ts(train, feature_cols)
val_x, val_y = structure_ts(val, feature_cols)
test_x, test_y = structure_ts(test, feature_cols)
test_x_ols, test_y_ols = structure_ts(test, features_2)

print("Saving")
with open("data/ts_train.pkl", "wb") as f:
    dump((train_x, train_y), f)

with open("data/ts_val.pkl", "wb") as f:
    dump((val_x, val_y), f)

with open("data/ts_test.pkl", "wb") as f:
    dump((test_x, test_y), f)

with open("data/ts_test_ols2.pkl", "wb") as f:
    dump((test_x_ols, test_y_ols), f)
