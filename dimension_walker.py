import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime
from yahoo_fin import stock_info as si
import pandas_datareader as pdr
import michigan_consumer_sentiment_index as mcsi


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
new_column_names = {'adjclose': 'Nasdaq'}
nasdaq_df.rename(columns=new_column_names, inplace=True)

snp_column_names = {'adjclose': 'S&P500'}
sp500_df.rename(columns=snp_column_names, inplace=True)

# 데이터프레임 합치기
merged_df = pd.merge(apple, nasdaq_df, on='Date', how='left')
merged_df = pd.merge(merged_df, sp500_df, on='Date', how='left')


apple = merged_df

# 원 달러 환율 데이터 가져오기
krwusd = yf.Ticker("KRW=X")

exchange_rate_data = krwusd.history(start=one_year_ago, end=current_date, auto_adjust=True)
exchange_rate_data['Date'] = pd.to_datetime(exchange_rate_data.index).strftime('%Y-%m-%d')
exchange_rate_data.set_index('Date', inplace=True)

# 인덱스(날짜)를 없애고 종가 열만 추출하여 Series로 변환
close_prices = exchange_rate_data["Close"]

# 위에서 생성한 close_prices 시리즈를 데이터프레임으로 변환
close_prices_df = close_prices.to_frame()
close_prices_df.rename(columns={'Close': 'Krwusd'}, inplace=True)

# 'Date' 열을 기준으로 두 데이터프레임 병합
close_prices_df.index = pd.to_datetime(close_prices_df.index)
apple = apple.merge(close_prices_df, left_on='Date', right_index=True, how='inner')

#국제유가 데이터 삽입
wti_oil = yf.Ticker("CL=F")  # CL=F는 WTI 원유 선물 계약의 티커입니다.
wti_oil_data = wti_oil.history(start=one_year_ago, end=current_date, auto_adjust=True)
wti_oil_data['Date'] = pd.to_datetime(wti_oil_data.index).strftime('%Y-%m-%d')
wti_oil_data.set_index('Date', inplace=True)

close_prices = wti_oil_data["Close"]

close_prices_df = close_prices.to_frame()
close_prices_df.rename(columns={'Close': 'Wti'}, inplace=True)

close_prices_df.index = pd.to_datetime(close_prices_df.index)
apple = apple.merge(close_prices_df, left_on='Date', right_index=True, how='inner')

# 금 가격 데이터 삽입
gold = yf.Ticker("GC=F")  # GC=F는 금 선물 계약의 티커입니다.
gold_data = gold.history(start=one_year_ago, end=current_date, auto_adjust=True)
gold_data['Date'] = pd.to_datetime(gold_data.index).strftime('%Y-%m-%d')
gold_data.set_index('Date', inplace=True)

close_prices = gold_data["Close"]

close_prices_df = close_prices.to_frame()
close_prices_df.rename(columns={'Close': 'Gold'}, inplace=True)

close_prices_df.index = pd.to_datetime(close_prices_df.index)
apple = apple.merge(close_prices_df, left_on='Date', right_index=True, how='inner')
print(apple.columns)

csi = mcsi.getCSI()
print(csi)