from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def mse(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error (MSE):", mse)
    return mse

def mae(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred )
    # print("Mean Absolute Error (MAE):", mae)
    return mae

def rmse(y_test, y_pred):
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse