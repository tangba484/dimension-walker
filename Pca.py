import util
import numpy as np
import pandas as pd
import Scaling
from pydmd import DMD
from util import make_sequence_dataset
import Model

def getPred(df,window_size=5,split=-80):
    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    target = pd.DataFrame(data , columns= ['Adj Close'])
    data = data.drop('Adj Close',axis = 1)

    n = 5
    df = util.Pca(data,n)

    df_np = df.to_numpy()
    target_np = target.to_numpy()

    X,Y = make_sequence_dataset(df_np , target_np , window_size)

    pca_pred = Model.lstm(X,Y,split=split)
    pca_pred = pca_pred*std + mean

    return pca_pred;