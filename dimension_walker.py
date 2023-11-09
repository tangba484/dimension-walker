import Stock
import Scaling
import Model , Pca , Raw , Dmd , Actual
import Evaluation
from util import make_sequence_dataset
import matplotlib.pyplot as plt

ticker = "aapl"
df = Stock.getStockDf(ticker)
df = Stock.addNasdaq(df)
df = Stock.addSnp500(df)
df = Stock.addKrwusd(df)
df = Stock.addWti(df)
df = Stock.addGold(df)
df = Stock.addCsi(df)



dmd_pred = Dmd.getPred(df)
pca_pred = Pca.getPred(df)
raw_pred = Raw.getPred(df)
y_test = Actual.getPred(df)
plt.figure(figsize=(12, 6))
plt.title('graph')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(dmd_pred, label='dmd_prediction')
plt.plot(raw_pred, label='raw_prediction')
plt.plot(pca_pred, label='pca_prediction')
plt.grid()
plt.legend(loc='best')

plt.show()




# M = 99999999
# l = []
# -80 117
# for split in range(-80,-300,-1):
#     y_test = Actual.getPred(df,split = split)
#     for window_size in range(10,380 + split - 1):
#         try:
#             dmd_pred = Dmd.getPred(df,window_size=window_size , split= split)
#             mae = Evaluation.Mae(y_test, dmd_pred)
#             if mae < M:
#                 l.append([split,window_size])
#                 M = mae
#         except:
#             continue