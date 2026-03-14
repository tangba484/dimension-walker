def add_features(df):

    # 이동 평균
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # 이동 평균 비율
    df["Close_MA5_ratio"] = df["Close"] / df["MA5"]
    df["Close_MA20_ratio"] = df["Close"] / df["MA20"]

    # 수익률
    df["Return1"] = df["Close"].pct_change(1)
    df["Return5"] = df["Close"].pct_change(5)
    df["Return10"] = df["Close"].pct_change(10)

    # 변동성
    df["Volatility5"] = df["Return1"].rolling(5).std()
    df["Volatility20"] = df["Return1"].rolling(20).std()

    # 거래량 이동평균
    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Band
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()

    df["BB_upper"] = rolling_mean + 2 * rolling_std
    df["BB_lower"] = rolling_mean - 2 * rolling_std

    # High-Low range
    df["HL_range"] = df["High"] - df["Low"]

    # Open-Close change
    df["OC_change"] = df["Close"] - df["Open"]

    df = df.dropna()

    return df