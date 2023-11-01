import Stock
import Scaling
import Model
import Evaluation
import Find_Show

ticker_list = []

with open("tickerList.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

for i in range(2, len(lines), 6):
    line = lines[i]
    parts = line.split("|")
    if len(parts) > 1:
        ticker = parts[0].strip()
        ticker_list.append(ticker)

for ticker in ticker_list:
    df = Stock.getStockDf(ticker)
    df = Stock.addNasdaq(df)
    df = Stock.addSnp500(df)
    df = Stock.addKrwusd(df)
    df = Stock.addWti(df)
    df = Stock.addGold(df)
    df = Stock.addCsi(df)

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA


    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    Find_Show.Find_Show(data , ticker , mean , std)

    
