import Stock
import Scaling


ticker = "tsla"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

df = df.drop(columns="Adj Close")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

result = Scaling.차분해야하는리스트반환(df)
data = Scaling.MinMaxScaling(Scaling.차분(df,result))

print(df)
print(data)
