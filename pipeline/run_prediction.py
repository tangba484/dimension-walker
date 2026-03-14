# import numpy as np
# from preprocessing.scaler import scale_data
# from preprocessing.dataset import make_dataset
# from preprocessing.pca import apply_pca
# from model.lstm_model import build_model
# from train.trainer import train
# from predict.predictor import predict
# from utils.gap_correction import correct_prediction_gap, correct_future_gap
# from config import horizon


# def run_prediction(df, use_pca=False, n_components=4):

#     mean = df["Close"].mean()
#     std = df["Close"].std()

#     scaled, scaler = scale_data(df)

#     X, y = make_dataset(scaled)

#     pca = None
#     if use_pca:
#         X, pca = apply_pca(X, n_components)

#     feature_dim = X.shape[2]

#     train_size = len(X) - 1

#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     model = build_model(feature_dim)

#     train(model, X_train, y_train)

#     pred_last_sample = predict(model, X_test)

#     pred_last_sample = pred_last_sample.flatten() * std + mean
#     y_test_scaled = y_test.flatten() * std + mean

#     pred_last_sample_corrected = correct_prediction_gap(
#         y_test_scaled[0],
#         pred_last_sample
#     )

#     # 미래 예측

#     model_future = build_model(feature_dim)

#     train(model_future, X, y)

#     latest_input = scaled[-200:]

#     if use_pca:

#         latest_input_pca = pca.transform(latest_input)

#         latest_input_pca = np.expand_dims(latest_input_pca, axis=0)

#         future_pred = predict(model_future, latest_input_pca)

#     else:

#         latest_input = np.expand_dims(latest_input, axis=0)

#         future_pred = predict(model_future, latest_input)

#     future_pred = future_pred.flatten() * std + mean

#     last_real_price = df["Close"].values[-1]

#     future_pred_corrected = correct_future_gap(
#         last_real_price,
#         future_pred
#     )

#     return y_test_scaled, pred_last_sample_corrected, future_pred_corrected


import numpy as np
from preprocessing.scaler import scale_data
from preprocessing.dataset import make_dataset
from preprocessing.pca import apply_pca
from model.lstm_model import build_model
from train.trainer import train
from predict.predictor import predict


def run_prediction(df, use_pca=False, n_components=4):
    mean = df["Close"].mean()
    std = df["Close"].std()

    scaled, scaler = scale_data(df)
    X, y = make_dataset(scaled)

    pca = None
    if use_pca:
        X, pca = apply_pca(X, n_components)

    feature_dim = X.shape[2]

    train_size = len(X) - 1
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 마지막 test 예측
    model = build_model(feature_dim)
    train(model, X_train, y_train)

    pred_last_sample = predict(model, X_test)

    pred_last_sample = pred_last_sample.flatten() * std + mean
    y_test_scaled = y_test.flatten() * std + mean

    actual_start = y_test_scaled[0]
    pred_start = pred_last_sample[0]
    gap = actual_start - pred_start
    pred_last_sample_corrected = pred_last_sample + gap

    # 미래 예측
    model_future = build_model(feature_dim)
    train(model_future, X, y)

    latest_input = scaled[-200:]

    if use_pca:
        latest_input_pca = pca.transform(latest_input)
        latest_input_pca = np.expand_dims(latest_input_pca, axis=0)
        future_pred = predict(model_future, latest_input_pca)
    else:
        latest_input = np.expand_dims(latest_input, axis=0)
        future_pred = predict(model_future, latest_input)

    future_pred = future_pred.flatten() * std + mean

    last_real_price = df["Close"].values[-1]
    future_gap = last_real_price - future_pred[0]
    future_pred_corrected = future_pred + future_gap

    return y_test_scaled, pred_last_sample_corrected, future_pred_corrected