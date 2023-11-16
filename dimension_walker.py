import Stock
import Scaling
import Pca , Raw , Dmd , Actual
import Evaluation
import matplotlib.pyplot as plt
import Model
import numpy as np
import os
import util
plt.switch_backend('agg')
raw = []
pca = []
dmd = []
ticker_list = util.getTikerList()
L = 0
for ticker in ticker_list:
    if ticker == "BRK-A":
        continue
    try:
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
        L += 1
        raw.append(raw_rmse)
        pca.append(pca_rmse)
        dmd.append(dmd_rmse)
        plt.text(len(y_test) + 0.1, dmd_pred[-1], f'DMD RMSE: {dmd_rmse}', color='red')
        plt.text(len(y_test) + 0.1, raw_pred[-1], f'Raw RMSE: {raw_rmse}', color='blue')
        plt.text(len(y_test) + 0.1, pca_pred[-1], f'PCA RMSE: {pca_rmse}', color='green')

        save_path = os.path.join(r'C:\Users\tangb\IdeaProjects\dimension-walker\images', f'{ticker}.png')
        plt.savefig(save_path)
    except:
        break
print("raw 평균:",sum(raw)/L,"pca 평균:",sum(pca)/L,"dmd 평균:",sum(dmd)/L)
print(len(raw),len(pca),len(dmd),L)

