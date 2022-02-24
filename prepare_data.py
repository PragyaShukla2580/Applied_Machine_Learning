import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib

def prepare(from_date,to_date):
    goog = yf.Ticker('goog')
    data = goog.history(start=from_date,end = to_date)
    print("The data looks like this: ")
    print(data.head())
    print("The length of the data is:")
    print(len(data))
    print("Columns in the dataset are:")
    print(data.columns)
    print(data.describe())
    print("Let's see if there are any null values")
    print(data.isnull().sum())
    print(data.info())
    joblib.dump(data, "df.joblib")

if __name__=="__main__":
    prepare("2010-01-01","2021-01-01")
