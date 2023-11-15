import util
import numpy as np
import pandas as pd
import Scaling
from util import make_sequence_dataset
import Model

def getPred(df):
    window_size = Model.window_size
    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    target = pd.DataFrame(data , columns= ['Adj Close'])
    raw_data = data.drop('Adj Close',axis = 1)

    target_np = target.to_numpy()
    raw_data_np = raw_data.to_numpy()

    X,Y = make_sequence_dataset(raw_data_np , target_np , window_size)
    raw_pred = Model.lstm(X,Y)

    raw_pred = raw_pred * std + mean

    return raw_pred