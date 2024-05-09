from tensorflow import keras
import pandas as pd
from lstm import LSTM
from pickle import load, dump
from typing import Callable
import matplotlib.pyplot as plt

# read data
with open("data/ts_train.pkl", "rb") as f:
    train_x, train_y = load(f)

with open("data/ts_val.pkl", "rb") as f:
    val_x, val_y = load(f)

feature_cols = train_x.shape[2]
window_size = train_x.shape[1]
# train_x/val_x/test_x are of shape [n_obs, n_timesteps, n_features]
# n_features is the only constant between train/val and each element of them

# make LSTM
model = LSTM(window_size, feature_cols)
model.compile(
    optimizer="adamw",
    loss="mean_squared_error",
    metrics=[
        "mean_absolute_error",
        "root_mean_squared_error",
        "r2_score",
    ],
)
# callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="checkpoint/LSTM.keras",
    monitor="val_loss",
    save_freq="epoch",
    save_best_only=True,
)
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    start_from_epoch=5,
)
lr_reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.8, patience=2, cooldown=1
)
#  train
history = model.fit(
    x=train_x,
    y=train_y,
    epochs=20,
    batch_size=1,
    validation_data=(val_x, val_y),
    callbacks=[checkpoint, es, lr_reduce],
)

with open("results/history.pkl", "wb") as f:
    dump(history, f)

# evaluate
with open("data/ts_test.pkl", "rb") as f:
    test_x, test_y = load(f)


output = model.evaluate(
    x=test_x,
    y=test_y,
    return_dict=True,
)
with open("results/output.pkl", "wb") as f:
    dump(output, f)


preds = model.predict(test_x)
with open("results/ts_test_preds.pkl", "wb") as f:
    dump(preds, f)
