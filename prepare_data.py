import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
from datetime import datetime
import yfinance as yf
import joblib
import pandas as pd
import dataframe_image as dfi



def prepare(from_date,to_date):
    data = DataReader('GOOG', data_source='yahoo', start=from_date, end=to_date)

    # print("The data looks like this: ")
    # print(data.head())
    dfi.export(data.head(), './prepare_data/data_head.png')

    # print("The length of the data is:")
    # print(len(data))
    with open('./prepare_data/length_of_dataset.txt', 'w') as f:
        f.write("Length of the dataset: "+ str(len(data)))

    # print("Columns in the dataset are:")
    # print(data.columns)
    df = pd.DataFrame(data.columns)
    df.to_csv('./prepare_data/data_columns.csv')

    # print(data.describe())
    with open('./prepare_data/columns_description.txt', 'w') as f:
        f.write(str(data.describe))

    # print("Let's see if there are any null values")
    # print(data.isnull().sum())
    df = pd.DataFrame(data.isnull().sum())
    df.to_csv('./prepare_data/data_null_values.csv')

def preprocess(from_date,to_date):
    # Scale the data
    # Get the stock quote
    data = DataReader('GOOG', data_source='yahoo', start=from_date, end=to_date)
    # Create a new dataframe with only the 'Close column
    data = data.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))


    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])


    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape


    joblib.dump(x_train, "./joblib/x_train.joblib")
    joblib.dump(y_train, "./joblib/y_train.joblib")
    joblib.dump(scaled_data, "./joblib/scaled_data.joblib")
    joblib.dump(data, "./joblib/data.joblib")
    joblib.dump(dataset, "./joblib/dataset.joblib")

if __name__=="__main__":

    prepare("2010-01-01","2021-01-01")
    preprocess("2010-01-01", "2021-01-01")
