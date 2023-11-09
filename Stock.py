from datetime import datetime, timedelta
import datetime
import yfinance as yf
import pandas as pd
from yahoo_fin import stock_info as si
import michigan_consumer_sentiment_index as mcsi

current_date = datetime.datetime.now()
one_year_ago = current_date - datetime.timedelta(days=380)

def getStockDf(stockTicker):

    df = yf.download(stockTicker, start=one_year_ago, end=current_date)

    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)

    df['SMA_10'] = df['Adj Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    return df

def addNasdaq(df):

    nasdaq_data = si.get_data('^IXIC', one_year_ago, current_date)

    nasdaq_data['Date'] = pd.to_datetime(nasdaq_data.index)
    nasdaq_data.set_index('Date', inplace=True)

    nasdaq_df = pd.DataFrame(nasdaq_data)
    
    nasdaq_df.drop(['open', 'high', 'low', 'close','volume', 'ticker'], axis=1, inplace=True)
    
    new_column_names = {'adjclose': 'Nasdaq'}
    nasdaq_df.rename(columns=new_column_names, inplace=True)


    merged_df = pd.merge(df, nasdaq_df, on='Date', how='left')

    return merged_df

def addSnp500(df):

    sp500_data = si.get_data('^GSPC', one_year_ago, current_date)

    sp500_data['Date'] = pd.to_datetime(sp500_data.index)  # 인덱스를 날짜 열로 설정
    sp500_data.set_index('Date', inplace=True)

    sp500_df = pd.DataFrame(sp500_data)
    
    sp500_df.drop(['open', 'high', 'low', 'close','volume', 'ticker'], axis=1, inplace=True)

    snp_column_names = {'adjclose': 'S&P500'}
    sp500_df.rename(columns=snp_column_names, inplace=True)

    merged_df = pd.merge(df, sp500_df, on='Date', how='left')
    return merged_df

def addKrwusd(df):
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
    df = df.merge(close_prices_df, left_on='Date', right_index=True, how='inner')
    return df

def addWti(df):
        #국제유가 데이터 삽입
    wti_oil = yf.Ticker("CL=F")  # CL=F는 WTI 원유 선물 계약의 티커입니다.
    wti_oil_data = wti_oil.history(start=one_year_ago, end=current_date, auto_adjust=True)
    wti_oil_data['Date'] = pd.to_datetime(wti_oil_data.index).strftime('%Y-%m-%d')
    wti_oil_data.set_index('Date', inplace=True)

    close_prices = wti_oil_data["Close"]

    close_prices_df = close_prices.to_frame()
    close_prices_df.rename(columns={'Close': 'Wti'}, inplace=True)

    close_prices_df.index = pd.to_datetime(close_prices_df.index)
    df = df.merge(close_prices_df, left_on='Date', right_index=True, how='inner')
    return df

def addGold(df):
    
    gold = yf.Ticker("GC=F") 
    gold_data = gold.history(start=one_year_ago, end=current_date, auto_adjust=True)
    gold_data['Date'] = pd.to_datetime(gold_data.index).strftime('%Y-%m-%d')
    gold_data.set_index('Date', inplace=True)

    close_prices = gold_data["Close"]

    close_prices_df = close_prices.to_frame()
    close_prices_df.rename(columns={'Close': 'Gold'}, inplace=True)

    close_prices_df.index = pd.to_datetime(close_prices_df.index)
    df = df.merge(close_prices_df, left_on='Date', right_index=True, how='inner')
    return df

def addCsi(df):
    
    csi = mcsi.getCSI()

    #  apple 데이터프레임의 인덱스 열을 리스트로 가져오기
    df_index_list = df.index.tolist()
    str_index_list = []

    # str_index_list에 formating(년-월)한 날짜 넣기
    for i in df_index_list:
        timestamp = pd.Timestamp(i)

        # Timestamp 객체를 문자열로 변환
        timestamp_str = timestamp.strftime('%Y-%m')
        
        str_index_list.append(timestamp_str)

    # 새로운 csi_df 만들어주기
    culumns_name = ['CSI']
    csi_df = pd.DataFrame(0, index=str_index_list, columns= culumns_name)
    csi_df['CSI'] = csi_df['CSI'].astype(float)

    # 날짜(전월)에 맞는 csi지수값 넣어주기
    for i in range(0,len(str_index_list)):
        
        index_value = str_index_list[i]
        string_to_split = index_value
        split_result = string_to_split.split('-')
        part1 = int(split_result[0])
        part2 = int(split_result[1])
        
        if part2 == 1:
            number_a = part1 - 1
            number_b = 12
        
        else:
            number_a = part1
            number_b = part2 - 1
            
        formatted_number_a = '{:02d}'.format(number_a)
        formatted_number_b = '{:02d}'.format(number_b)
        
        index_value = formatted_number_a + '-' + formatted_number_b
        csi_df.at[str_index_list[i], 'CSI'] = csi[index_value]

        
    csi_df['Date'] = df_index_list
    csi_df.set_index('Date', inplace=True)
    df = pd.concat([df, csi_df], axis=1)
    return df
