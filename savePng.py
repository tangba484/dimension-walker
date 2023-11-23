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

        num_none_values = len(y_test) + Model.split

        raw_pred =  np.pad(Raw.getPred(df), ((num_none_values, 0), (0, 0)), mode='constant', constant_values=None)
        dmd_pred = np.pad(Dmd.getPred(df), ((num_none_values, 0), (0, 0)), mode='constant', constant_values=None)
        pca_pred = np.pad(Pca.getPred(df), ((num_none_values, 0), (0, 0)), mode='constant', constant_values=None)

        average_pred = (raw_pred + dmd_pred + pca_pred) / 3

        rp_pred = (raw_pred + pca_pred) / 2

        print(len(raw_pred),num_none_values,len(y_test))
        plt.figure(figsize=(14, 6))
        plt.title('graph')
        plt.ylabel('adj close')
        plt.xlabel('period')
        plt.plot(y_test, label='actual')
        
        plt.plot( raw_pred, label='raw_prediction',color='blue')
        plt.plot(dmd_pred, label='dmd_prediction',color='red')
        plt.plot(pca_pred, label='pca_prediction',color='green')
        plt.plot(average_pred,label='average_pred',color = 'orange')
        plt.plot(rp_pred,label='rp_pred',color = 'purple')
        plt.grid()
        plt.legend(loc='best')

# raw_rmse = round(Evaluation.rmse(y_test[Model.split:], raw_pred[Model.split:]), 2)
# pca_rmse = round(Evaluation.rmse(y_test[Model.split:], pca_pred[Model.split:]), 2)
# dmd_rmse = round(Evaluation.rmse(y_test[Model.split:], dmd_pred[Model.split:]), 2)

# plt.text(len(y_test) + 0.1, raw_pred[-1], f'Raw RMSE: {raw_rmse}', color='blue')
# plt.text(len(y_test) + 0.1, dmd_pred[-1], f'DMD RMSE: {dmd_rmse}', color='red')
# plt.text(len(y_test) + 0.1, pca_pred[-1], f'PCA RMSE: {pca_rmse}', color='green')

        # raw_rmse = round(Evaluation.rmse(y_test[Model.split:], raw_pred[Model.split:]), 2)
        # pca_rmse = round(Evaluation.rmse(y_test[Model.split:], pca_pred[Model.split:]), 2)
        # dmd_rmse = round(Evaluation.rmse(y_test[Model.split:], dmd_pred[Model.split:]), 2)

        # plt.text(len(y_test) + 0.1, raw_pred[-1], f'Raw RMSE: {raw_rmse}', color='blue')
        # plt.text(len(y_test) + 0.1, dmd_pred[-1], f'DMD RMSE: {dmd_rmse}', color='red')
        # plt.text(len(y_test) + 0.1, pca_pred[-1], f'PCA RMSE: {pca_rmse}', color='green')

        # L += 1
        
        # raw.append(raw_rmse)
        # pca.append(pca_rmse)
        # dmd.append(dmd_rmse)

        save_path = os.path.join(r'C:\Users\tangb\IdeaProjects\dimension-walker\average', f'{ticker}.png')
        plt.savefig(save_path)
    except:
        break
print("raw 평균:",sum(raw)/L,"pca 평균:",sum(pca)/L,"dmd 평균:",sum(dmd)/L)