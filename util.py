import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def HeatMap(left_df , target):
    target = target.reset_index()

    left_df = pd.concat([left_df, target['Adj Close']], axis=1)
    print(left_df)
    # 예시 데이터 생성
    # 상관관계 행렬 계산
    correlation_matrix = left_df.corr()

    # Heatmap으로 상관관계 시각화
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")

    # 플롯 설정 (선택 사항)
    plt.title("상관관계 Heatmap")
    plt.show()
    return

def Pca(df,n):
    n_components = n
    pca = PCA(n_components=n_components)
    df = pca.fit_transform(df)

    pca_data = pca.fit_transform(df)
    df = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    return df

def make_sequence_dataset(feature , label , window_size = 20):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list) , np.array(label_list)

def getTikerList():
    ticker_list = []
    with open("tickerList.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i in range(2, len(lines), 6):
        line = lines[i]
        parts = line.split("|")
        if len(parts) > 1:
            ticker = parts[0].strip()
            ticker_list.append(ticker)
    return ticker_list;