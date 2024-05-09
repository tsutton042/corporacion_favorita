"""
File that contains shared functions
"""


import numpy as np
import pandas as pd


def lin_interp_step(n_days: int, start_val: float, end_val: float) -> list:
    if n_days > 0:
        diff = end_val - start_val
        step = diff / (n_days + 1)
        return [start_val + step * (i + 1) for i in range(n_days)]
    else:
        return []


def lin_interp(oil_prices: pd.Series) -> pd.Series:
    # code like as we assume that all entries are neighbouring days (true in the data)
    prices = []
    days_na = 0
    init_val = oil_prices.iloc[0] if oil_prices.notna().iloc[0] else oil_prices.mean()
    for day_price in oil_prices:
        if np.isnan(day_price):
            days_na += 1
        else:
            interp_vals = lin_interp_step(days_na, init_val, day_price)
            prices.extend(interp_vals)
            prices.append(day_price)
            init_val = day_price
            days_na = 0
    oil_price_interp = pd.Series(prices, index=oil_prices.index)
    return oil_price_interp


def interp_oil(train: pd.DataFrame, oil_price_interp: pd.Series) -> pd.DataFrame:
    for date in oil_price_interp.index:
        idxs = train[train.date == date].index
        train.loc[idxs, "dcoilwtico"] = oil_price_interp[date]
    return train
