import Stock
import Scaling

df = Stock.getStockDf('AAPL')
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

print(df)

print(Scaling.RobustScaling(df))