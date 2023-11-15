import Stock
import Scaling
import Pca , Raw , Dmd , Actual
import Evaluation
import matplotlib.pyplot as plt
import Model
import numpy as np


ticker = "aapl"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

y_test = Actual.getPred(df)
dmd_pred = Dmd.getPred(df)
raw_pred = Raw.getPred(df)
pca_pred = Pca.getPred(df)
plt.figure(figsize=(14, 6))
plt.title('graph')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(dmd_pred, label='dmd_prediction',color='red')
plt.plot(raw_pred, label='raw_prediction',color='blue')
plt.plot(pca_pred, label='pca_prediction',color='green')
plt.grid()
plt.legend(loc='best')

raw_rmse = round(Evaluation.rmse(y_test, raw_pred), 2)
pca_rmse = round(Evaluation.rmse(y_test, pca_pred), 2)
dmd_rmse = round(Evaluation.rmse(y_test, dmd_pred), 2)

plt.text(len(y_test) + 0.1, dmd_pred[-1], f'DMD RMSE: {dmd_rmse}', color='red')
plt.text(len(y_test) + 0.1, raw_pred[-1], f'Raw RMSE: {raw_rmse}', color='blue')
plt.text(len(y_test) + 0.1, pca_pred[-1], f'PCA RMSE: {pca_rmse}', color='green')


plt.show()