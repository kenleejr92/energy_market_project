__author__ = 'kenlee'
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Data_Scaler(object):

    def __init__(self, method):
        if method == 'standard':
            self.scaler = preprocessing.StandardScaler()
        if method == 'normalize':
            self.scaler = preprocessing.Normalizer()
        if method == 'min_max':
            self.scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        if method == 'max_abs':
            self.scaler = preprocessing.MaxAbsScaler()
        if method == 'robust':
            self.scaler = preprocessing.RobustScaler()



    def scale_training_data(self, df, cols):
        df[cols] = self.scaler.fit_transform(df[cols])
        return df

    def scale_testing_data(self, df, cols):
        df[cols] = self.scaler.transform(df[cols])
        return df

    def inverse_scale(self, df, cols):
        df[cols] = self.scaler.inverse_transform(df[cols])
        return df

def reject_outliers(data, m=3):
        mean = np.mean(data)
        std = np.std(data)
        for i,d in enumerate(data):
            if abs(d - mean) >= m*std:
                data[i] = m*std
        return data