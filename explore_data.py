import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
import dataframe_image as dfi
from pandas_datareader.data import DataReader
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
from bokeh.plotting import figure, output_file, show

def explore():

    # Get the stock quote
    data = DataReader('GOOG', data_source='yahoo', start='2010-01-01', end="2021-01-01")
    data['Open'].plot(label = 'Google', figsize = (15,7))
    # plt.title('Stock Price of Google')
    plt.savefig('./Explore/Stock_Price.png')



    # stock fluctuation
    data['Volume'].plot(label='Google', figsize=(15, 7))
    # plt.title('Volume of Stock traded')
    # plt.legend()
    plt.savefig('./Explore/Volume_of_Stock_Traded.png')

    # Market Capitalisation
    data['MarktCap'] = data['Open'] * data['Volume']
    data['MarktCap'].plot(label='Google', figsize=(15, 7))
    # plt.title('Market Cap')
    # plt.legend()
    plt.savefig('./Explore/Market_Cap.png')



if __name__ == "__main__":
    explore()