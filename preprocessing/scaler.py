import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data(df):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)

    return scaled, scaler


