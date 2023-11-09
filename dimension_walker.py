import Stock
import Scaling
import Model
import Evaluation

ticker = "aapl"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pydmd import DMD

# result = Scaling.차분해야하는리스트반환(df)
# data = Scaling.표준화(Scaling.차분(df,result))
mean = np.mean(df['Adj Close'])
std = np.std(df['Adj Close'])
data = Scaling.표준화(df)
raw_data = data
answer = pd.DataFrame(data , columns= ['Adj Close'])

data = data.drop('Adj Close',axis = 1)
n_components = 5
pca = PCA(n_components=n_components)
print(data)
pca_data = pca.fit_transform(data)
print(pca_data,"@@@")


df = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])
print(df)

# import note
# note.HeatMap(df , answer)
df_np = df.to_numpy()
print(df_np)

data_array = data.values

# NumPy 배열을 reshape하여 다차원 배열로 변환
reshaped_data = data_array.reshape(len(answer["Adj Close"]), 14)  # 예: 3행 3열로 변환

dmd = DMD(svd_rank=3)  # svd_rank는 주요한 모드의 개수를 지정합니다


# DMD 모델 훈련
dmd.fit(data_array)

# DMD 결과 얻기
modes = dmd.modes  # DMD 모드
eigs = dmd.eigs  # 고유값


answer_np = answer.to_numpy()


window_size = 10 # 변수
def make_sequence_dataset(feature , label , window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list) , np.array(label_list)

X,Y = make_sequence_dataset(df_np , answer_np , window_size)

pca_pred = Model.lstm(X,Y)
y_test = Y[-120:]  # test counts

X,Y = make_sequence_dataset(raw_data.to_numpy() , answer_np , window_size)
raw_pred = Model.lstm(X,Y)

X,Y = make_sequence_dataset(modes , answer_np , window_size)
DMD_pred = Model.lstm(X,Y)


plt.figure(figsize=(12, 6))
plt.title('graph')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pca_pred, label='pca_prediction')
plt.plot(raw_pred, label='raw_prediction')
plt.plot(DMD_pred, label='DMD_prediction')
plt.grid()
plt.legend(loc='best')

plt.show()

print('PCA_pred MAE score : ',Evaluation.Mae(y_test , pca_pred))
print('PCA_pred MSE score : ',Evaluation.Mse(y_test , pca_pred))

print('raw_pred MAE score : ',Evaluation.Mae(y_test , raw_pred))
print('raw_pred MSE score : ',Evaluation.Mse(y_test , raw_pred))

print('DMD_pred MAE score : ',Evaluation.Mae(y_test , DMD_pred))
print('DMD_pred MSE score : ',Evaluation.Mse(y_test , DMD_pred))