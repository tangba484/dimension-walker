import numpy as np
from config import horizon
def make_dataset(data, window=200):

    X = []
    y = []

    for i in range(len(data) - window - horizon):

        X.append(data[i:i+window])

        y.append(data[i+window:i+window+horizon,3])
    return np.array(X), np.array(y)