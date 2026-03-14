from data.loader import load_stock
from data.add_features import add_features

from pipeline.run_prediction import run_prediction
from evaluation.metrics import calculate_metrics
from evaluation.plot import plot_results

ticker = "AAPL"

df = load_stock(ticker)

df = add_features(df)

y_true, pred_test_no_pca, future_no_pca = run_prediction(df, use_pca=False)

_, pred_test_pca, future_pca = run_prediction(df, use_pca=True)

metrics = calculate_metrics(
    y_true,
    pred_test_no_pca,
    pred_test_pca
)

plot_results(
    df,
    y_true,
    pred_test_no_pca,
    future_no_pca,
    pred_test_pca,
    future_pca,
    metrics,
    ticker
)