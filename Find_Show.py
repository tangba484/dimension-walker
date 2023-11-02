import Stock
import Scaling
import Model
import Evaluation
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout , LSTM, Input

def Find_Show(df,ticker , mean , std):
    plt.switch_backend('agg')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    raw_data = df
    answer = pd.DataFrame(df , columns= ['Adj Close'])

    df = df.drop('Adj Close',axis = 1)

    mae_list = []

    for k in range(10,13):
        n_components = k
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(df)


        df_pca = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])

        df_np = df_pca.to_numpy()
        answer_np = answer.to_numpy()

        window_size = 20
    
        X,Y = util.make_sequence_dataset(df_np , answer_np , window_size)

        pca_pred = Model.lstm(X,Y)
        y_test = Y[-200:]

        X,Y = util.make_sequence_dataset(raw_data.to_numpy() , answer_np , window_size)
        raw_pred = Model.lstm(X,Y)
        
        mae_list.append(Evaluation.Mae(y_test , pca_pred))
       
    min_value = min(mae_list)
    min_index = mae_list.index(min_value)
        
    n_components = int(min_index) + 1 + 9
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)


    df_pca = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])

    df_np = df_pca.to_numpy()
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
    plt.title(f' ticker = {ticker}, n = {n_components} \n (pca - actual) mae = {Evaluation.Mae(pca_pred , y_test)} \n (pca - raw) mae = {Evaluation.Mae(raw_pred , pca_pred)}')
    plt.ylabel('adj close')
    plt.xlabel('period')
    plt.plot(y_test, label='actual')
    plt.plot(pca_pred, label='pca_pred')
    plt.plot(raw_pred, label='raw_pred')
    plt.grid()
    plt.legend(loc='best')

    file_path = f'images/{ticker}.png'
    plt.savefig(file_path)

    plt.close()

    