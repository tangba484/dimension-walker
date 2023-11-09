import util
import numpy as np
import pandas as pd
import Scaling
import Model

def getPred(df,window_size=5,split = 1):


    mean = np.mean(df['Adj Close'])
    std = np.std(df['Adj Close'])
    data = Scaling.표준화(df)

    target = pd.DataFrame(data , columns= ['Adj Close'])
    target_np = target.to_numpy()

    y_test = target_np[Model.split:]

    y_test = y_test * std + mean

    return y_test
