import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime
from yahoo_fin import stock_info as si
import pandas_datareader as pdr


# 한글 폰트 설정 (예: 나눔고딕 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'

# 데이터 불러오기 (예: Apple 주식 데이터)
# 여기에서는 yfinance 라이브러리를 사용하여 데이터를 불러올 수 있습니다.

import yfinance as yf

# datetime 모듈 사용해서 현재 날싸 불러오기

current_date = datetime.datetime.now()
one_year_ago = current_date - datetime.timedelta(days=365)



apple = yf.download('AAPL', start=one_year_ago, end=current_date)


# 데이터 전처리
apple['Date'] = pd.to_datetime(apple.index)  # 인덱스를 날짜 열로 설정
apple.set_index('Date', inplace=True)

# 특성 엔지니어링: 이동평균선을 포함한 기술적 지표를 계산할 수 있습니다.

apple['SMA_10'] = apple['Adj Close'].rolling(window=10).mean()
apple['SMA_50'] = apple['Adj Close'].rolling(window=50).mean()
apple['Returns'] = apple['Adj Close'].pct_change()
apple = apple.dropna()

#나스닥 지수 데이터 가져오기
nasdaq_data = si.get_data('^IXIC', one_year_ago, current_date)


# S&P 500 지수 데이터 가져오기
sp500_data = si.get_data('^GSPC', one_year_ago, current_date)

nasdaq_data['Date'] = pd.to_datetime(nasdaq_data.index)  # 인덱스를 날짜 열로 설정
nasdaq_data.set_index('Date', inplace=True)

sp500_data['Date'] = pd.to_datetime(sp500_data.index)  # 인덱스를 날짜 열로 설정
sp500_data.set_index('Date', inplace=True)

# 데이터를 DataFrame으로 변환
nasdaq_df = pd.DataFrame(nasdaq_data)
sp500_df = pd.DataFrame(sp500_data)





# 나스닥과 s&p500 지수에서 불필요한 열 제거
nasdaq_df.drop(['open', 'high', 'low', 'close','volume', 'ticker'], axis=1, inplace=True)
sp500_df.drop(['open', 'high', 'low', 'close','volume', 'ticker'], axis=1, inplace=True)

# 열 이름 변경
new_column_names = {'adjclose': 'nasdaq'}
nasdaq_df.rename(columns=new_column_names, inplace=True)

snp_column_names = {'adjclose': 'sp500'}
sp500_df.rename(columns=snp_column_names, inplace=True)

# 데이터프레임 합치기
merged_df = pd.merge(apple, nasdaq_df, on='Date', how='left')
merged_df = pd.merge(merged_df, sp500_df, on='Date', how='left')


apple = merged_df

print(apple)




