from sklearn.metrics import confusion_matrix
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np


from sklearn.preprocessing import MinMaxScaler
x_train = joblib.load('./joblib/x_train.joblib')
y_train = joblib.load('./joblib/y_train.joblib')
y_test= joblib.load("./Results/y_test.joblib")

scaled_data = joblib.load('./joblib/scaled_data.joblib')
data = joblib.load('./joblib/data.joblib')
rmse = joblib.load('./Results/rmse.joblib')
predictions = joblib.load('./Results/predictions.joblib')
training_data_len = int(np.ceil(len(data) * .95))
scaler = MinMaxScaler(feature_range=(0,1))
dataset = joblib.load('./joblib/dataset.joblib')
scaled_data = scaler.fit_transform(dataset)
prediction = joblib.load("./Results/prediction.joblib")

def evaluate():


    errors = mean_squared_error(y_test, prediction, squared=False)
    with open('./Results/Root_Mean_Squared_Error.txt', 'w') as f:
        f.write("Root Mean Squared Error: "+ str(errors))

    # print(errors)




if __name__ == "__main__":
    evaluate()

