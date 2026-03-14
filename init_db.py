import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "stock.db")

os.makedirs(INSTANCE_DIR, exist_ok=True)

print("INIT DB PATH:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS predicted_prices")
cur.execute("DROP TABLE IF EXISTS actual_prices")
cur.execute("DROP TABLE IF EXISTS prediction_runs")

cur.execute("""
CREATE TABLE prediction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    created_at TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    n_components INTEGER,
    mae_no_pca REAL,
    rmse_no_pca REAL,
    mae_pca REAL,
    rmse_pca REAL
)
""")

cur.execute("""
CREATE TABLE actual_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    date TEXT NOT NULL,
    close_price REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES prediction_runs(id)
)
""")

cur.execute("""
CREATE TABLE predicted_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_type TEXT NOT NULL,
    segment_type TEXT NOT NULL,
    step INTEGER NOT NULL,
    price REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES prediction_runs(id)
)
""")

conn.commit()
conn.close()

print("DB 초기화 완료")