import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import MinMaxScaler


def 차분해야하는리스트반환(df):
    non_stationary_list = []
    columnLength = len(df.columns)
    for i in range(columnLength):
        if df.columns[i] == "CSI":
            continue
        result = kpss(df.iloc[:, i], regression= 'c')
        if result[1] > 0.05:
            continue
        else:
            non_stationary_list.append(df.columns[i])
    
    return non_stationary_list

def 차분안하는리스트반환(df):
    stationary_list = []
    columnLength = len(df.columns)
    for i in range(columnLength):
        print(kpss(df.iloc[:, i], regression= 'c'))
        result = kpss(df.iloc[:, i], regression= 'c')
        
        if result[1] > 0.05:
            stationary_list.append(df.columns[i])
    return stationary_list
    
    
def 차분(df,non_stationary_list):
    for i in non_stationary_list:
        differenced_prices = df[i].diff().dropna()
        df[i] = differenced_prices
    
    return df[1:]
    
def 표준화(df):
    mean = df.mean()
    std_dev = df.std()
    standardized_data = (df - mean) / std_dev
    return standardized_data

def MinMaxScaling(df):

    column_names = df.columns.tolist()

    mMscaler = MinMaxScaler()
    mMscaler.fit(df)

    mMscaled_data = mMscaler.transform(df)

    mMscaled_data = pd.DataFrame(mMscaled_data, columns=column_names)

    return mMscaled_data
    
def MaxAbsScaling(df):
    scaler = MaxAbsScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)    
    return scaled_df

def RobustScaling(df):   
    scaler = RobustScaler()    
    scaled_data = scaler.fit_transform(df)    
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)   
    return scaled_df