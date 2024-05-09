from tensorflow import keras


class LSTM:
    def __init__(self, window_size, n_features):
        ipts = keras.layers.Input(
            shape=[window_size, n_features]
        )  # (batch_size, n_timesteps, n_features)
        l = keras.layers.LSTM(n_features)(
            ipts
        )  # 8 features seems reasonable. default args look good!
        x = keras.layers.Dense(max(4, n_features // 2), activation="relu")(
            l
        )  # do this to increase capacity
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=ipts, outputs=outputs)
        self.summary()

    def summary(self):
        print(self.model.summary())

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.model.predict(*args, **kwargs)


if __name__ == "__main__":
    LSTM(2)
    LSTM(4)
    LSTM(8)
    LSTM(10)
    LSTM(12)
