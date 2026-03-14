import numpy as np
import matplotlib.pyplot as plt
from config import horizon


def plot_results(df,
                 y_true,
                 pred_test_no_pca,
                 future_no_pca,
                 pred_test_pca,
                 future_pca,
                 metrics,
                 ticker):

    close_prices = df["Close"].values[-200:]

    plt.figure(figsize=(14, 7))

    plt.plot(
        np.arange(len(close_prices)),
        close_prices,
        label="Actual Close",
        color="blue",
        linewidth=2
    )

    test_prediction_index = np.arange(len(close_prices) - horizon - 1, len(close_prices) - 1)

    future_index = np.arange(len(close_prices) - 1, len(close_prices) + horizon - 1)

    plt.plot(
        test_prediction_index,
        pred_test_no_pca,
        label="Test Prediction (No PCA)",
        color="red"
    )

    plt.plot(
        future_index,
        future_no_pca,
        label="Future Prediction (No PCA)",
        color="green"
    )

    plt.plot(
        test_prediction_index,
        pred_test_pca,
        label="Test Prediction (PCA)",
        color="orange"
    )

    plt.plot(
        future_index,
        future_pca,
        label="Future Prediction (PCA)",
        color="purple"
    )

    textstr = (
        f"No PCA\n"
        f"MAE  : {metrics['mae_no_pca']:.2f}\n"
        f"RMSE : {metrics['rmse_no_pca']:.2f}\n\n"
        f"PCA\n"
        f"MAE  : {metrics['mae_pca']:.2f}\n"
        f"RMSE : {metrics['rmse_pca']:.2f}"
    )

    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.title(f"{ticker.upper()} Close Price / PCA vs No PCA Prediction Comparison")

    plt.legend()
    plt.grid(True)
    plt.show()