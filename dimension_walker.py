import Stock
import Scaling

ticker = "005930.KS"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)

print(df)

print(Scaling.표준화(df))

result = Scaling.차분해야하는리스트반환(df)
print(Scaling.표준화(Scaling.차분(df,result)))
