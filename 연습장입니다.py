import Stock
import Scaling
import Model
import Evaluation
import numpy as np
from pydmd import DMD



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


data = Scaling.MinMaxScaling(df)
raw_data = data
answer = pd.DataFrame(data , columns= ['Adj Close'])

data = data.drop('Adj Close',axis = 1)

data_array = data.values

# NumPy 배열을 reshape하여 다차원 배열로 변환
reshaped_data = data_array.reshape(297, 14)  # 예: 3행 3열로 변환

dmd = DMD(svd_rank=3)  # svd_rank는 주요한 모드의 개수를 지정합니다



# DMD 모델 훈련
dmd.fit(data_array)

# DMD 결과 얻기
modes = dmd.modes  # DMD 모드
eigs = dmd.eigs  # 고유값

# 결과 출력

print(modes)
print(eigs)
max_value = max(eigs)
max_index = max(eigs)

print(modes[:,max_index])

