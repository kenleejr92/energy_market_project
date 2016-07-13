# coding: utf-8

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
        if model_type == "A":
            for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    features.append([work_day_or_holiday(dt),
                                          dt.hour,
                                          dt.weekday()]
                                          + dflzhub.iloc[pred_hour_index - 2*24:pred_hour_index - 1*24].tolist()
                                          + dflzhub.iloc[pred_hour_index - 7*24:pred_hour_index - 6*24].tolist())
            feature_labels = ['Holiday', 'Hour', 'Day']\
                             + [('P(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                             + [('P(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]
            numerical_features = ['Hour']\
                                 + [('P(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                                 + [('P(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]
            idx_wout_1st_week = list(dflzhub.index.values)[7*24:]
            features_df = pd.DataFrame(data=features,
                                       index=idx_wout_1st_week,
                                       columns=feature_labels)
            min_max_scale(features_df, numerical_features)
            features_df = encode_onehot(features_df, 'Day')
            #normalize numerical values

            return features_df.join(dflzhub, how='left')

        if model_type == 'B':
            for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    features.append([work_day_or_holiday(dt),
                                          dt.hour,
                                          dt.weekday()]
                                          + dflzhub.iloc[pred_hour_index - 2*24:pred_hour_index - 1*24].tolist()
                                          + dflzhub.iloc[pred_hour_index - 7*24:pred_hour_index - 6*24].tolist()
                                          + load_df.iloc[pred_hour_index - 2*24:pred_hour_index - 1*24].tolist()
                                          + load_df.iloc[pred_hour_index - 7*24:pred_hour_index - 6*24].tolist())
            feature_labels = ['Holiday', 'Hour', 'Day']\
                             + [('P(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                             + [('P(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]\
                             + [('L(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                             + [('L(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]
            numerical_features = ['Hour']\
                                 + [('P(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                                 + [('P(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]\
                                 + [('L(h-%s)' % str(i+1)) for i in range(24, 48)][::-1]\
                                 + [('L(h-%s)' % str(i+1)) for i in range(144, 168)][::-1]
            idx_wout_1st_week = list(dflzhub.index.values)[7*24:]
            features_df = pd.DataFrame(data=features,
                                       index=idx_wout_1st_week,
                                       columns=feature_labels)
            min_max_scale(features_df, numerical_features)
            features_df = encode_onehot(features_df, 'Day')
            return features_df.join(dflzhub, how='left')

    
   
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
def train_test_validate(ft, train_size=0.6, val_size=0.2, test_size=0.2):
    feature_target = ft.as_matrix()
    num_features = feature_target.shape[1]

    train_indices = []
    test_indices = []
    val_indices = []
    for i in range(MONTHS_PER_YEAR):
        train_i, test_i, val_i = sample_month(i, train_size, val_size, test_size)
        train_indices = train_indices + train_i
        test_indices =  test_indices + test_i
        val_indices = val_indices + val_i
    train_indices = [i*HRS_PER_DAY for i in train_indices]
    test_indices = [i*HRS_PER_DAY for i in test_indices]
    val_indices = [i*HRS_PER_DAY for i in val_indices]
    train_set = np.zeros((1, num_features))
    test_set = np.zeros((1, num_features))
    val_set = np.zeros((1, num_features))
    for i in train_indices:
        train_set = np.concatenate((train_set, feature_target[i:i+HRS_PER_DAY, :]), axis=0)
    for i in test_indices:
        test_set = np.concatenate((test_set, feature_target[i:i+HRS_PER_DAY, :]), axis=0)
    for i in val_indices:
        val_set = np.concatenate((val_set, feature_target[i:i+HRS_PER_DAY, :]), axis=0)
    train_set = (train_set[1:, 0:num_features-1], train_set[1:, num_features-1])
    val_set = (val_set[1:, 0:num_features-1], val_set[1:, num_features-1])
    test_set = (test_set[1:, 0:num_features-1], test_set[1:, num_features-1])
    return train_set, val_set, test_set

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

def min_max_scale(df,cols):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[cols] = min_max_scaler.fit_transform(df[cols])

def sample_month(month_index, train_size,test_size,val_size):
    np.random.seed(1111)
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

def test_Query_DAM_by_SP():
    qdsp = Feature_Processor()
    qdsp.query("2012-01-01", "2012-12-31")
    qdsp.construct_feature_vector_matrix('LZ_NORTH', 'B')


if __name__ == '__main__':
    test_Query_DAM_by_SP()
