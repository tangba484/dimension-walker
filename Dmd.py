import numpy as np
import pandas as pd
import Scaling
from pydmd import DMD
from util import make_sequence_dataset
import Model

def getPred(df):
    window_size = Model.window_size
    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    target = pd.DataFrame(data , columns= ['Adj Close'])
    data = data.drop('Adj Close',axis = 1)

    data_array = data.values

    dmd = DMD(svd_rank=5)
    
    dmd.fit(data_array)

    modes = dmd.modes
    eigs = dmd.eigs

    target_np = target.to_numpy()
    X,Y = make_sequence_dataset(modes , target_np , window_size)
    Dmd_pred = Model.lstm(X,Y)
    Dmd_pred = Dmd_pred * std + mean
    # Dmd_pred = np.roll(Dmd_pred, -2)

    return Dmd_pred
