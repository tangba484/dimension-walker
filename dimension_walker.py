import Stock
import Scaling

df = Stock.getStockDf('AAPL')
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

result = Scaling.차분해야하는리스트반환(df)

print(Scaling.표준화(Scaling.차분(df, result)))