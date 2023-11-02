import Stock
import Scaling
import Model
import Evaluation
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import make_sequence_dataset

ticker = "tsla"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

mean = np.mean(df['Adj Close'])
std = np.std(df['Adj Close'])
data = Scaling.표준화(df)
raw_data = data
target = pd.DataFrame(data , columns= ['Adj Close'])
data = data.drop('Adj Close',axis = 1)
predictions = [0]*16
for n in range(1,14):
    df = util.Pca(data , n)
    df_np = df.to_numpy()
    target_np = target.to_numpy()
    X,Y = make_sequence_dataset(df_np , target_np )
    pca_pred = Model.lstm(X,Y)
    pca_pred = pca_pred*std + mean
    predictions[n] = pca_pred

plt.figure(figsize=(12, 6)) 
y_test = Y[-200:]
y_test = y_test*std + mean
X,Y = make_sequence_dataset(raw_data.to_numpy() , target_np)
raw_pred = Model.lstm(X,Y)*std + mean
for n in range(7, 14):
    plt.subplot(3, 5, n)
    plt.plot(predictions[n], label=f'n={n}')  # 예측 결과 그래프를 그립니다.
    plt.plot(y_test , label = 'actual')
    plt.plot(raw_pred , label = 'raw_pred')
    value =Evaluation.Mae(y_test, raw_pred) - Evaluation.Mae(y_test , predictions[n])
    plt.text(10, value,f'value{value}', fontsize=12, color='red')
    plt.plot()
    plt.xlabel('Time')
    plt.ylabel('Prediction')
    plt.legend()

plt.tight_layout()
plt.show()