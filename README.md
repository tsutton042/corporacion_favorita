# Corporation Favorita

Contains various models that solve the (Corporation Favorita)[https://www.kaggle.com/c/favorita-grocery-sales-forecasting/overview] Kaggle competition

# Modelling ideas
* [X] Moving average - run & evaluate with valid prediction structure
* [X] Persistance models - run & evaluate with valid prediction structure
    * [x] Pre-defined horizon/window
    * [ ] Dynamic horizon/window
* [ ] ARMA/ARIMA
* [ ] ANNs
    * [X] LSTMs (Again, poor $R^2$, value appeared to be of different signs for training and val sets?)
        * [ ] Combine with Convolution layer on outputs?
    * [ ] CNNs?
    * [ ] Autoencoder? May be good to enforce consitency with other labels
* [X] Simpler models
    * [X] Linear regression (EDA - poor $R^2$)
    * [X] RandomForest (EDA - poor $R^2$)
    * [ ] NN-based modelling


# Data issues
* Train takes too much compute to read in
    * Have taken a sample of the last ~3.2% (4,000,000) of the rows
    * Anything much bigger causes the OOM killer to be invoked
    * Could write a data reader that chunks the data if this becomes a big issue/performance bottleneck
