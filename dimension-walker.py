import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 한글 폰트 설정 (예: 나눔고딕 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'

# 데이터 불러오기 (예: Apple 주식 데이터)
# 여기에서는 yfinance 라이브러리를 사용하여 데이터를 불러올 수 있습니다.

import yfinance as yf
apple = yf.download('AAPL', start='2023-01-01', end='2023-9-15')

print(apple.columns)
# 데이터 전처리
apple['Date'] = pd.to_datetime(apple.index)  # 인덱스를 날짜 열로 설정
apple.set_index('Date', inplace=True)

# 특성 엔지니어링: 이동평균선을 포함한 기술적 지표를 계산할 수 있습니다.

apple['SMA_10'] = apple['Adj Close'].rolling(window=10).mean()
apple['SMA_50'] = apple['Adj Close'].rolling(window=50).mean()
apple['Returns'] = apple['Adj Close'].pct_change()
apple = apple.dropna()

# 종속 변수와 독립 변수 설정
X = apple[['SMA_10', 'SMA_50', 'Returns']]
y = apple['Adj Close']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'Training R-squared score: {train_score}')
print(f'Test R-squared score: {test_score}')

# 주가 예측
last_data = X[-1:].values
predicted_price = model.predict(last_data)

print(f'다음 날 Apple 주식 종가 예측: {predicted_price[0]}')

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(apple.index, apple['Adj Close'], label='실제 주가')
plt.axvline(x=apple.index[-1], color='r', linestyle='--', linewidth=2, label='예측 시작일')
plt.plot(apple.index[-1], predicted_price[0], marker='o', markersize=8, color='g', label='예측 주가')
plt.xlabel('날짜')
plt.ylabel('주가')
plt.legend()
plt.show()
