from sklearn.metrics import mean_squared_error, mean_absolute_error

def Mse(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error (MSE):", mse)
    return mse

def Mae(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred )
    # print("Mean Absolute Error (MAE):", mae)
    return mae