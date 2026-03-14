from data.loader import load_stock
from data.add_features import add_features
from pipeline.run_prediction import run_prediction
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import window,horizon
from services.db_service import (
    get_latest_run_by_ticker,
    create_prediction_run,
    save_actual_prices,
    save_predicted_series,
    load_chart_data,
    get_all_saved_tickers,
    delete_runs_by_ticker,
)

WINDOW_SIZE = window
HORIZON = horizon
N_COMPONENTS = 4

def generate_and_save_prediction(ticker):
    ticker = ticker.upper()

    df = load_stock(ticker)
    df = add_features(df)

    # 최근 200일 실제 종가/날짜
    actual_df = df[["Close"]].tail(WINDOW_SIZE).copy()
    actual_dates = df.index[-WINDOW_SIZE:]
    actual_prices = actual_df["Close"].values

    # No PCA
    y_test_true, pred_test_no_pca, future_pred_no_pca = run_prediction(
        df,
        use_pca=False
    )

    # PCA
    _, pred_test_pca, future_pred_pca = run_prediction(
        df,
        use_pca=True,
        n_components=N_COMPONENTS
    )

    # 성능지표 계산
    mae_no_pca = mean_absolute_error(y_test_true, pred_test_no_pca)
    rmse_no_pca = np.sqrt(mean_squared_error(y_test_true, pred_test_no_pca))

    mae_pca = mean_absolute_error(y_test_true, pred_test_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_test_true, pred_test_pca))


    # 실행 기록 저장
    run_id = create_prediction_run(
        ticker=ticker,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        n_components=N_COMPONENTS,
        mae_no_pca=mae_no_pca,
        rmse_no_pca=rmse_no_pca,
        mae_pca=mae_pca,
        rmse_pca=rmse_pca
    )

    # 실제 종가 200일 저장
    save_actual_prices(
        run_id=run_id,
        dates=actual_dates,
        close_prices=actual_prices
    )

    # 예측값 저장
    save_predicted_series(run_id, "no_pca", "test", pred_test_no_pca)
    save_predicted_series(run_id, "no_pca", "future", future_pred_no_pca)
    save_predicted_series(run_id, "pca", "test", pred_test_pca)
    save_predicted_series(run_id, "pca", "future", future_pred_pca)

    return run_id


def get_or_create_chart_data(ticker):
    ticker = ticker.upper()

    latest_run = get_latest_run_by_ticker(ticker)

    if latest_run is None:
        run_id = generate_and_save_prediction(ticker)
    else:
        run_id = latest_run["id"]

    return load_chart_data(run_id)


def refresh_ticker_prediction(ticker):
    ticker = ticker.upper()

    delete_runs_by_ticker(ticker)

    return generate_and_save_prediction(ticker)


def refresh_all_saved_tickers():
    tickers = get_all_saved_tickers()

    if not tickers:
        print("갱신할 ticker가 없습니다.")
        return

    print("갱신 대상:", tickers)

    for ticker in tickers:
        try:
            print(f"[갱신 시작] {ticker}")
            refresh_ticker_prediction(ticker)
            print(f"[갱신 완료] {ticker}")
        except Exception as e:
            print(f"[갱신 실패] {ticker}: {e}")
