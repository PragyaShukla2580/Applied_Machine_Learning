import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import train

# load the model from disk
# model = joblib.load('./model/model.joblib')

x_train = joblib.load('./joblib/x_train.joblib')
y_train = joblib.load('./joblib/y_train.joblib')
scaled_data = joblib.load('./joblib/scaled_data.joblib')
data = joblib.load('./joblib/data.joblib')
rmse = joblib.load('Assignment_1/Results/rmse.joblib')
predictions = joblib.load('Assignment_1/Results/predictions.joblib')
training_data_len = int(np.ceil(len(data) * .95))
scaler = MinMaxScaler(feature_range=(0,1))
dataset = joblib.load('./joblib/dataset.joblib')
scaled_data = scaler.fit_transform(dataset)

def score():

    print(rmse)

    # Plot the data

    # X_scaled = scaler.fit_transform(predictions)
    # obj = scaler.fit(predictions)

    prediction = scaler.inverse_transform(predictions)


    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = prediction
    # Visualize the data
    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('./Prediction/Predictions.png')
    joblib.dump(prediction, "Assignment_1/Results/prediction.joblib")


if __name__=="__main__":
    score()