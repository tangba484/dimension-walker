import yfinance as yf
from flask import render_template
def load_stock(ticker):
    df = yf.download(ticker, period="3y")
    df.columns = df.columns.get_level_values(0) 

    df = df[['Open','High','Low','Close','Volume']]

    df = df.dropna()

    return df