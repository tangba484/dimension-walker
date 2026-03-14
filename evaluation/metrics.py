import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, pred_no_pca, pred_pca):

    mae_no_pca = mean_absolute_error(y_true, pred_no_pca)
    rmse_no_pca = np.sqrt(mean_squared_error(y_true, pred_no_pca))

    mae_pca = mean_absolute_error(y_true, pred_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_true, pred_pca))

    return {
        "mae_no_pca": mae_no_pca,
        "rmse_no_pca": rmse_no_pca,
        "mae_pca": mae_pca,
        "rmse_pca": rmse_pca
    }