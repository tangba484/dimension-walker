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

# result = Scaling.차분해야하는리스트반환(df)
# data = Scaling.표준화(Scaling.차분(df,result))
mean = np.mean(df['Adj Close'])
std = np.std(df['Adj Close'])
data = Scaling.표준화(df)
raw_data = data
answer = pd.DataFrame(data , columns= ['Adj Close'])

data = data.drop('Adj Close',axis = 1)
n_components = 10
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(data)

pca_result = pca.fit_transform(pca_data)
df = pd.DataFrame(data=pca_result, columns=[f'PC{i}' for i in range(1, n_components + 1)])

# import note
# note.HeatMap(df , answer)
df_np = df.to_numpy()
answer_np = answer.to_numpy()


window_size = 20
def make_sequence_dataset(feature , label , window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list) , np.array(label_list)

X,Y = make_sequence_dataset(df_np , answer_np , window_size)

pca_pred = Model.lstm(X,Y)
y_test = Y[-200:]

X,Y = make_sequence_dataset(raw_data.to_numpy() , answer_np , window_size)
raw_pred = Model.lstm(X,Y)

y_test = y_test*std + mean
pca_pred = pca_pred*std + mean
raw_pred = raw_pred*std + mean

plt.figure(figsize=(12, 6))
plt.title('graph')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pca_pred, label='pca_prediction')
plt.plot(raw_pred, label='raw_prediction')
plt.grid()
plt.legend(loc='best')

plt.show()
print(df)
Evaluation.Mae(y_test , pca_pred)
Evaluation.Mse(y_test , pca_pred)

Evaluation.Mae(y_test , raw_pred)
Evaluation.Mse(y_test , raw_pred)