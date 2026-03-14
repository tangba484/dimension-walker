import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "instance", "stock.db")


def get_db_connection():
    print("DB_PATH:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_run_by_ticker(ticker):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT *
        FROM prediction_runs
        WHERE ticker = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (ticker.upper(),)
    ).fetchone()
    conn.close()
    return row


def create_prediction_run(
    ticker,
    window_size,
    horizon,
    n_components=None,
    mae_no_pca=None,
    rmse_no_pca=None,
    mae_pca=None,
    rmse_pca=None
):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO prediction_runs (
            ticker, created_at, window_size, horizon, n_components,
            mae_no_pca, rmse_no_pca, mae_pca, rmse_pca
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker.upper(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            window_size,
            horizon,
            n_components,
            mae_no_pca,
            rmse_no_pca,
            mae_pca,
            rmse_pca
        )
    )

    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_actual_prices(run_id, dates, close_prices):
    conn = get_db_connection()
    cur = conn.cursor()

    rows = []
    for idx, (date, price) in enumerate(zip(dates, close_prices)):
        rows.append((run_id, idx, str(date), float(price)))

    cur.executemany(
        """
        INSERT INTO actual_prices (run_id, idx, date, close_price)
        VALUES (?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()
    conn.close()


def save_predicted_series(run_id, model_type, segment_type, values):
    conn = get_db_connection()
    cur = conn.cursor()

    rows = []
    for step, price in enumerate(values):
        rows.append(
            (run_id, model_type, segment_type, step, float(price))
        )

    cur.executemany(
        """
        INSERT INTO predicted_prices (run_id, model_type, segment_type, step, price)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()
    conn.close()


def load_chart_data(run_id):
    conn = get_db_connection()

    actual_rows = conn.execute(
        """
        SELECT idx, date, close_price
        FROM actual_prices
        WHERE run_id = ?
        ORDER BY idx
        """,
        (run_id,)
    ).fetchall()

    pred_rows = conn.execute(
        """
        SELECT model_type, segment_type, step, price
        FROM predicted_prices
        WHERE run_id = ?
        ORDER BY model_type, segment_type, step
        """,
        (run_id,)
    ).fetchall()

    run_info = conn.execute(
        """
        SELECT *
        FROM prediction_runs
        WHERE id = ?
        """,
        (run_id,)
    ).fetchone()

    conn.close()
    
    dates = [str(row["date"])[:10] for row in actual_rows]
    actual_prices = [row["close_price"] for row in actual_rows]

    data = {
        "run_info": run_info,
        "dates": dates,
        "actual_prices": actual_prices,
        "no_pca_test": [],
        "no_pca_future": [],
        "pca_test": [],
        "pca_future": []
    }

    for row in pred_rows:
        key = f"{row['model_type']}_{row['segment_type']}"
        data[key].append(row["price"])

    return data


def get_all_saved_tickers():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM prediction_runs
        ORDER BY ticker
        """
    ).fetchall()
    conn.close()

    return [row["ticker"] for row in rows]



def delete_runs_by_ticker(ticker):
    conn = get_db_connection()
    cur = conn.cursor()

    run_rows = cur.execute(
        """
        SELECT id
        FROM prediction_runs
        WHERE ticker = ?
        """,
        (ticker.upper(),)
    ).fetchall()

    run_ids = [row["id"] for row in run_rows]

    if run_ids:
        placeholders = ",".join(["?"] * len(run_ids))

        cur.execute(
            f"""
            DELETE FROM predicted_prices
            WHERE run_id IN ({placeholders})
            """,
            run_ids
        )

        cur.execute(
            f"""
            DELETE FROM actual_prices
            WHERE run_id IN ({placeholders})
            """,
            run_ids
        )

        cur.execute(
            """
            DELETE FROM prediction_runs
            WHERE ticker = ?
            """,
            (ticker.upper(),)
        )

    conn.commit()
    conn.close()
