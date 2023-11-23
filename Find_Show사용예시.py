import Stock
import Scaling
import Model
import Evaluation
import Find_Show
import numpy as np
import util
ticker_list = util.getTikerList();

for ticker in ticker_list:
    df = Stock.getStockDf(ticker)
    df = Stock.addNasdaq(df)
    df = Stock.addSnp500(df)
    df = Stock.addKrwusd(df)
    df = Stock.addWti(df)
    df = Stock.addGold(df)
    df = Stock.addCsi(df)

    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    Find_Show.Find_Show(data , ticker , mean , std)

    
