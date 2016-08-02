
# coding: utf-8

# In[51]:



import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MySQLdb
from MySQLdb.constants import FIELD_TYPE
from datetime import datetime
import calendar
import holidays
import random
from sklearn import cross_validation
from sklearn import datasets
from sklearn import preprocessing
from Query_ERCOT_DB import Query_ERCOT_DB

MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30
HRS_PER_DAY = 24

# Acquire DAM SPP for all settlement points for a specific date range
class Feature_Processor(Query_ERCOT_DB):
    # list of settlement points is common across all instances of the Feature_Processor class

    table_headers = []
    Query_ERCOT_DB.c.execute("""SHOW COLUMNS FROM DAM_SPPs""")
    r = list(Query_ERCOT_DB.c.fetchall())
    for sp in r:
        if sp[0] == "delivery_date" or sp[0] == "hour_ending":
            continue
        table_headers.append(sp[0])
    Query_ERCOT_DB.c.execute("""SHOW COLUMNS FROM Load_by_LZ""")
    r = list(Query_ERCOT_DB.c.fetchall())
    for sp in r:
        if sp[0] == "delivery_date" or sp[0] == "hour_ending":
            continue
        table_headers.append(sp[0])
    '''
    Query the list of settlement points and remove the heading "Settlement Point"
    self.start_date - start date of query
    self.end_date - end date of query
    self.dts - list of date_time objects in the query result
    self.df - pandas data frame representing the query result
    '''
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.dts = None
        self.df = None
        self.features_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.standard_scaler = None


    '''
    Query for all prices for all load zones and hubs for specified date range
    Creates a pandas data frame of the query
    '''
    def query(self, sd, ed):
        self.start_date = sd
        self.end_date = ed
        Query_ERCOT_DB.c.execute("""SELECT * FROM DAM_SPPs
                INNER JOIN Load_by_LZ
                USING (delivery_date,hour_ending)
                WHERE DAM_SPPs.delivery_date > "%s"
                AND DAM_SPPs.delivery_date < "%s"
                ORDER BY DAM_SPPs.delivery_date, DAM_SPPs.hour_ending""" % (sd, ed))
        result = list(Query_ERCOT_DB.c.fetchall())
        fresult = []
        for r in result:
            temp = ()
            date = r[0]
            time = str(int(r[1].split(":")[0])-1)
            dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
            for x in r[2:]:
                temp = temp + (float(x),)
            r = (dt,) + temp
            fresult.append(r)
        self.df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=self.table_headers)
        self.dts = self.df.index

    '''
    Given a load zone or hub, creates a feature data frame for the specified model
    Model A (Benchmark):
        Input1: Day-Type indicator
        Input2: Hour indicator
        Input3: Holiday indicator
        Input4: Hourly price of day d-1
        Input5: Hourly price of day d-7
    Model B (exogenous variables):
    '''
    def construct_feature_vector_matrix(self, lzhub, model_type):
        dflzhub = self.df[lzhub + '_SPP']
        load_df = self.df[lzhub + '_load']
        features = []
        feature_labels = None
        numerical_features = None
        idx_wout_1st_week = None
        
        if model_type == "A":
            for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    features.append([work_day_or_holiday(dt),
                                          dt.hour,
                                          dt.weekday(),
                                          dt.month,
                                          dflzhub.iloc[pred_hour_index - 1*24],
                                          dflzhub.iloc[pred_hour_index - 7*24]])
            feature_labels = ['Holiday', 'Hour', 'Day', 'Month', 'P(h-24)', 'P(h-168)']
            idx_wout_1st_week = list(dflzhub.index.values)[7*24:]

        if model_type == 'B':
            for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    features.append([work_day_or_holiday(dt),
                                          dt.hour,
                                          dt.weekday(),
                                          dt.month,
                                          dflzhub.iloc[pred_hour_index - 24],
                                          dflzhub.iloc[pred_hour_index - 25],
                                          dflzhub.iloc[pred_hour_index - 47],
                                          dflzhub.iloc[pred_hour_index - 48],
                                          dflzhub.iloc[pred_hour_index - 72],
                                          dflzhub.iloc[pred_hour_index - 120],
                                          dflzhub.iloc[pred_hour_index - 144],
                                          dflzhub.iloc[pred_hour_index - 167],
                                          dflzhub.iloc[pred_hour_index - 168]])
            feature_labels = ['Holiday', 'Hour', 'Day', 'Month', 'P(h-24)', 'P(h-25)', 'P(h-47)', 'P(h-48)', 'P(h-72)',                              'P(h-120)', 'P(h-144)', 'P(h-167)', 'P(h-168)']
            idx_wout_1st_week = list(dflzhub.index.values)[7*24:]

        if model_type == 'C':
            for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    features.append([work_day_or_holiday(dt),
                                          dt.hour,
                                          dt.weekday(),
                                          dt.month,
                                          load_df.iloc[pred_hour_index],
                                          dflzhub.iloc[pred_hour_index - 1*24],
                                          dflzhub.iloc[pred_hour_index - 7*24],
                                          load_df.iloc[pred_hour_index - 1*24],
                                          load_df.iloc[pred_hour_index - 7*24]])
            feature_labels = ['Holiday', 'Hour', 'Day', 'Month', 'FLoad', 'P(h-24)', 'P(h-168)', 'L(h-24)', 'L(h-168)']
            idx_wout_1st_week = list(dflzhub.index.values)[7*24:]
        # features dataframe before one-hot encoding and normalization of numerical features
        self.features_df = pd.DataFrame(data=features,
                                   index=idx_wout_1st_week,
                                   columns=feature_labels)
    
        self.features_df = encode_onehot(self.features_df, 'Day')
        self.features_df = encode_onehot(self.features_df, 'Month')
        self.features_df = encode_onehot(self.features_df, 'Hour')
        self.features_df = self.features_df.join(dflzhub, how='left')
        return self.features_df
    

    def scale_num_features(self, lzhub, model, df, method, set_type = 'Train'):
        modelA_features = ['P(h-24)', 'P(h-168)']
        modelB_features = ['P(h-24)', 'P(h-25)', 'P(h-47)', 'P(h-48)', 'P(h-72)', 'P(h-120)', 'P(h-144)', 'P(h-167)', 'P(h-168)']
        modelC_features = ['FLoad', 'P(h-24)', 'P(h-168)', 'L(h-24)', 'L(h-168)']
        numerical_features = []
        if model == 'A':
            numerical_features = modelA_features[:]
        if model == 'B':
            numerical_features = modelB_features[:]
        if model == 'C':
            numerical_features = modelC_features[:]
        numerical_features.append(lzhub + '_SPP')

            
        if method == 'L2_norm':
            output_norm = np.linalg.norm(df[lzhub + '_SPP'].as_matrix())
            df = normalize(df, numerical_features)
            return df, output_norm
        
        if method == 'standard_scale':
            df = self.standard_scale(df, numerical_features, set_type)
            return df
        
        if method == 'robust_scale':
            df = robust_scale(df, numerical_features)
            return df
        if method == 'max_abs_scale':
            df = max_abs_scale(df, numerical_features)
            return df
   
    '''
    Plots the prices for all load zones and hubs for the specified date range
    '''
    def plot(self):
        self.df.plot()
        plt.title("SPP by LZ and HUB for %s" % self.start_date.split("-")[0])
        plt.xlabel("Date-Time")
        plt.ylabel("SPP")
        plt.legend()
        plt.show()

    def get_settlement_points(self):
        return self.table_headers

    '''
    Splits the feature data frame into train, test, and validation sets
    Performs seasonal sampling; splits the date range into months and then samples within each month without replacement
        60% of each month for training
        20% of each month for validation
        20% of each month for testing
    '''
    def train_test_validate(self, train_size=0.6, test_size=0.2):
        ft = self.features_df
        train_indices = []
        test_indices = []
        val_indices = []
        for i in range(MONTHS_PER_YEAR):
            train_i, test_i, val_i = sample_month(i, train_size, test_size)
            train_indices = train_indices + train_i
            test_indices = test_indices + test_i
            val_indices = val_indices + val_i
        train_indices = [i*HRS_PER_DAY for i in train_indices]
        test_indices = [i*HRS_PER_DAY for i in test_indices]
        val_indices = [i*HRS_PER_DAY for i in val_indices]
        train_dfs = []
        test_dfs = []
        val_dfs = []
        for i in train_indices:
            train_dfs.append(ft.iloc[i:i+HRS_PER_DAY])
        self.train_df = pd.concat(train_dfs)
        for i in test_indices:
            test_dfs.append(ft.iloc[i:i+HRS_PER_DAY])
        self.test_df = pd.concat(test_dfs)
        for i in val_indices:
            val_dfs.append(ft.iloc[i:i+HRS_PER_DAY])
        self.val_df = pd.concat(val_dfs)
        return self.train_df, self.val_df, self.test_df
    
    def convert_dfs_to_numpy(self,df):
        num_features = df.as_matrix().shape[1]
        return (df.ix[:, 0:num_features-1].as_matrix(), df.ix[:, num_features-1].as_matrix()) 

    def standard_scale(self, df, cols, set_type):
        if set_type == 'Train':
            self.standard_scaler = preprocessing.StandardScaler()
            df[cols] = self.standard_scaler.fit_transform(df[cols])
        else:
            df[cols] = self.standard_scaler.transform(df[cols])
        return df

    def inverse_standard_scale(self, lzhub, model, df):
        modelA_features = ['P(h-24)', 'P(h-168)']
        modelB_features = ['P(h-24)', 'P(h-25)', 'P(h-47)', 'P(h-48)', 'P(h-72)', 'P(h-120)', 'P(h-144)', 'P(h-167)', 'P(h-168)']
        modelC_features = ['FLoad', 'P(h-24)', 'P(h-168)', 'L(h-24)', 'L(h-168)']
        numerical_features = []
        if model == 'A':
            numerical_features = modelA_features[:]
        if model == 'B':
            numerical_features = modelB_features[:]
        if model == 'C':
            numerical_features = modelC_features[:]
        numerical_features.append(lzhub + '_SPP')
        print(numerical_features)
        return self.standard_scaler.inverse_transform(df[numerical_features])

def string_to_date(string_date):
    return datetime.strptime(string_date, "%Y-%m-%d %H")

def date_to_string(date):
    return date.strftime("%Y-%m-%d %H")

def weekday_of_date(date):
    return calendar.day_name[date.weekday()]

def work_day_or_holiday(date):
    us_holidays = holidays.UnitedStates()
    if date in us_holidays or weekday_of_date(date) == "Sunday" or weekday_of_date(date) == "Saturday":
        return int(1)
    else: return int(0)
    
def encode_onehot(df, cols):
    enc = preprocessing.OneHotEncoder()
    index = df[cols].index
    data = enc.fit_transform(df[cols].reshape(-1, 1)).toarray()
    one_hot_df = pd.DataFrame(data=data, index=index, columns=[cols + '%s' % i for i in range(data.shape[1])])
    del df[cols]
    return df.join(one_hot_df, how='inner')

def max_abs_scale(df,cols):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    df[cols] = max_abs_scaler.fit_transform(df[cols])
    return df

def min_max_scale(df,cols):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[cols] = min_max_scaler.fit_transform(df[cols])
    return df



def robust_scale(df,cols):
    rscaler = preprocessing.RobustScaler()
    df[cols] = rscaler.fit_transform(df[cols])
    return df

def normalize(df,cols):
    normalizer = preprocessing.Normalizer()
    df[cols] = normalizer.fit_transform(df[cols])
    return df

def sample_month(month_index, train_size, test_size):
    np.random.seed(22943)
    indices = np.arange(0, DAYS_PER_MONTH)
    set_indices = set(indices)
    train_indices = np.random.choice(indices,
                                  int(DAYS_PER_MONTH*train_size),
                                  replace=False).tolist()
    test_indices = np.random.choice(list(set_indices.difference(set(train_indices))),
                                 int(DAYS_PER_MONTH*test_size),
                                 replace=False).tolist()
    val_indices = list(set_indices.difference(set(train_indices)).difference(test_indices))

    train_indices = [i + month_index*DAYS_PER_MONTH for i in train_indices]
    test_indices = [i + month_index*DAYS_PER_MONTH for i in test_indices]
    val_indices = [i + month_index*DAYS_PER_MONTH for i in val_indices]
    return train_indices, test_indices, val_indices





