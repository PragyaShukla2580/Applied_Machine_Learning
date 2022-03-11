from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import logging
import time
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

try:
    x_train = joblib.load('./joblib/x_train.joblib')
    y_train = joblib.load('./joblib/y_train.joblib')
    scaled_data = joblib.load('./joblib/scaled_data.joblib')
    data = joblib.load('./joblib/data.joblib')
    dataset = joblib.load('./joblib/dataset.joblib')
    training_data_len = int(np.ceil(len(data) * .95))
    scaler = MinMaxScaler(feature_range=(0,1))
except Exception:
    logging.error("Error in reading file")


def training():
    """
                    Description
                    ----------
                    This function is designed for training. The model stays here and we have
                    used LSTM model for training."""

    # Build the LSTM model
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(hidden_size2, return_sequences=False))
    model.add(Dense(num_layers))
    model.add(Dense(1))

    # Compile the model
    try:
        model.compile(optimizer='adam', loss='mean_squared_error')
    except Exception:
        logging.critical("Model not compiled")
        return

    # Train the model
    try:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs)
    except Exception:
        logging.critical("Model not fitted")
        return
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    logging.info("Training done.")
    # predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    print(rmse)

    joblib.dump(rmse, "./Results/rmse.joblib")
    joblib.dump(predictions, "./Results/predictions.joblib")
    joblib.dump(y_test, "./Results/y_test.joblib")




if __name__=="__main__":

    try:
        f = open('configs/params.json', )
        data = json.load(f)
    except Exception:
        logging.error("Couldn't start")

    hidden_size1 = int(data["model_configs_train"]["hidden_size1"])
    hidden_size2 = int(data["model_configs_train"]["hidden_size2"])
    num_layers = int(data["model_configs_train"]["num_layers"])
    num_epochs = int(data["model_configs_train"]["num_epochs"])
    batch_size = int(data["model_configs_train"]["batch_size"])

    training()