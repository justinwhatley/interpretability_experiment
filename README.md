# Interpretability Experiment


<!-- ABOUT THE PROJECT -->
## About The Project

This project explores different interpretability strategies to interpret tabular data at scale. In particular, it explores applications of ShAP to gradient boosting and NN models. 

Project aim:
* Compare feature extraction from black-box models to classical linear approaches
* Compare black-box model interpretations, seeing where these converge or diverge in meaningful ways
* Apply out-of-memory strategies that will enable both batched training of models, followed by model interpretation to unpack predictive components of the inputs


<!-- ROADMAP -->
## Roadmap

Experiments: 
1. data_exploration.ipynb:  Data exploration using Dask. Here statistics are gathered across the entire dataset to gain some intuition about the data and test Dask functionality

2. lr_interpretability.ipynb:  Uses Dask functionality for out-of-memory training of a linear regression model with weight extraction as a proxy for feature importance. Here only continuous variables were taken as inputs as a test, but this may be expand to use categorical inputs with some kind of categorical encoding like OHE and take them in too.

3. lightgbm_interpretablity.ipynb:  Uses a gradient boosting strategy with lightgbm and normal in-memory processing of a pandas dataframe to apply ShAP for model interpretability. To scale this, will have to add a preprocessing step for gradual ingestion of the data instead of depending on in-memory process. 

4. lightgbm_dask.ipynb: Checks compatibility with lightgbm with Dask. Not fully supported (or I'm missing something!), this will need more work. 

5. nn_interpretability: (in progress) Uses a feed-forward neural network to model the data, representing categoricals as embeddings. Uses ShAP for interpretability of model - will see how this handles encoded categorical data. 


