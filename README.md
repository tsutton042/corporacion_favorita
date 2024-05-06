# Corporation Favorita

Contains various models that solve the (Corporation Favorita)[https://www.kaggle.com/c/favorita-grocery-sales-forecasting/overview] Kaggle competition

# Modelling ideas
* [x] Moving average
* [x] Persistance models
    * [x] Pre-defined horizon/window
    * [ ] Dynamic horizon/window
* [ ] ARMA/ARIMA
* [ ] ANNs
    * [ ] LSTMs
        * [ ] Combine with Convolution layer on outputs?
    * [ ] CNNs
    * [ ] Autoencoder? May be good to enforce consitency with other labels
* [ ] Simpler models
    * [ ] Linear regression
    * [ ] RandomForest/XGBoost
    * [ ] NN-based modelling


# Data issues
* Train takes too much compute to read in
    * Have taken a sample of the last ~3.2% (4,000,000) of the rows
    * Anything much bigger causes the OOM killer to be invoked
    * Could write a data reader that chunks the data if this becomes a big issue/performance bottleneck
