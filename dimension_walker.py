import Stock
import Scaling

ticker = "AAPL"
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
data = Scaling.표준화(Scaling.차분(df,result))
print(df)
print(data)
n_components = 5 
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(data)


pca_result = pca.fit_transform(pca_data)
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i}' for i in range(1, n_components + 1)])

print(pca_df)