
# coding: utf-8

# In[51]:



import operator
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
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from Query_ERCOT_DB import Query_ERCOT_DB
from Data_Scaler import Data_Scaler
from Exogeneous_Variables import Exogeneous_Variables
import os

from Query_CRR import Query_CRR
import scipy.signal
import math

MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 28
HRS_PER_DAY = 24

# Acquire DAM SPP for all settlement points for a specific date range
class Feature_Processor(Query_ERCOT_DB):
    # list of settlement points is common across all instances of the Feature_Processor class

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
        self.data_scaler = None
        self.numerical_features = None
        self.lzhub = None
        self.wind_df = None
        self.load_df = None
        self.table_headers = []
        Query_ERCOT_DB.c.execute("""SHOW COLUMNS FROM DAM_SPPs""")
        r = list(Query_ERCOT_DB.c.fetchall())
        for sp in r:
            if sp[0] == "delivery_date" or sp[0] == "hour_ending":
                continue
            self.table_headers.append(sp[0])

    '''
    Query for all prices for all load zones and hubs for specified date range
    Creates a pandas data frame of the query
    '''
    def query(self, sd, ed):
        self.start_date = sd
        self.end_date = ed
        Query_ERCOT_DB.c.execute("""SELECT * FROM DAM_SPPs
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
    def construct_feature_vector_matrix(self, lzhub, model_type='B'):
        self.lzhub = lzhub
        prices = self.df[[lzhub]]
        zone = 'WGRPP_' + lzhub[3:]
        EV = Exogeneous_Variables()
        self.wind_df = EV.query_wind(self.start_date, self.end_date)[[zone]]
        self.load_df = EV.query_load(self.start_date, self.end_date)[['SystemTotal']]
        price_wind_load_df = prices.join(self.wind_df, how='inner').join(self.load_df, how='inner')
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        price_wind_load_df[['SystemTotal', zone]] = imp.fit_transform(price_wind_load_df[['SystemTotal', zone]])
        dflzhub = price_wind_load_df[lzhub]
        wind = price_wind_load_df[zone]
        load = price_wind_load_df['SystemTotal']
        features = []
        feature_labels = None
        idx_wout_1st_week = None

        if model_type == 'A':
            for dt, series in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if type(pred_hour_index) == slice: continue
                if pred_hour_index - 96 >= 0:
                    categorical_features = np.array([work_day_or_holiday(dt), dt.hour, dt.weekday(), dt.month])
                    past_spps = np.array([dflzhub.iloc[pred_hour_index-24],
                                          dflzhub.iloc[pred_hour_index-48],
                                          dflzhub.iloc[pred_hour_index-72],
                                          dflzhub.iloc[pred_hour_index-96]])
                    past_values = np.concatenate([categorical_features,
                                                 past_spps])
                    features.append(past_values)

            feature_labels = ['Holiday', 'Hour', 'Day', 'Month'] + \
                             ['P(h-24)', 'P(h-48)', 'P(h-72)', 'P(h-96)']


            self.numerical_features = ['P(h-24)', 'P(h-48)', 'P(h-72)', 'P(h-96)'] + \
                                      [lzhub]
            # added 18 to 96 for coding error?
            idx_wout_1st_week = list(dflzhub.index.values)[96:]

        if model_type == 'B':
            for dt, series in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if type(pred_hour_index) == slice: continue
                if pred_hour_index - 96 >= 0:
                    categorical_features = np.array([work_day_or_holiday(dt), dt.hour, dt.weekday(), dt.month])
                    past_spps = np.array([dflzhub.iloc[pred_hour_index-24],
                                          dflzhub.iloc[pred_hour_index-48],
                                          dflzhub.iloc[pred_hour_index-72],
                                          dflzhub.iloc[pred_hour_index-96]])
                    past_wind = np.array([wind.iloc[pred_hour_index]])
                    past_load = np.array([load.iloc[pred_hour_index]])
                    past_values = np.concatenate([categorical_features,
                                                 past_spps, past_wind, past_load])
                    features.append(past_values)

            feature_labels = ['Holiday', 'Hour', 'Day', 'Month'] + \
                             ['P(h-24)', 'P(h-48)', 'P(h-72)', 'P(h-96)'] + \
                             ['W(h)'] + \
                             ['L(h)']

            self.numerical_features = ['P(h-24)', 'P(h-48)', 'P(h-72)', 'P(h-96)'] + \
                                      ['W(h)'] + \
                                      ['L(h)'] + \
                                      [lzhub]
            idx_wout_1st_week = list(dflzhub.index.values)[96:]

        self.features_df = pd.DataFrame(data=features,
                                   index=idx_wout_1st_week,
                                   columns=feature_labels)


        self.features_df = encode_onehot(self.features_df, 'Day')
        self.features_df = encode_onehot(self.features_df, 'Month')
        self.features_df = encode_onehot(self.features_df, 'Hour')
        self.features_df = self.features_df.join(dflzhub, how='left')

        return self.features_df
    

    def inverse_scale_testing(self):
        return self.data_scaler.inverse_scale(self.test_df, self.numerical_features)[self.lzhub].as_matrix()

    def inverse_scale_validation(self):
        return self.data_scaler.inverse_scale(self.val_df, self.numerical_features)[self.lzhub].as_matrix()

    def inverse_scale_prediction(self, y_pred):
        test_copy = pd.DataFrame.copy(self.test_df)
        test_copy[self.lzhub] = y_pred
        return self.data_scaler.inverse_scale(test_copy, self.numerical_features)[self.lzhub].as_matrix()


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
    def train_test_validate(self, scaling, method='sequential', train_size=0.6, test_size=0.2):
        self.data_scaler = Data_Scaler(scaling)
        ft = self.features_df
        train_indices = []
        test_indices = []
        val_indices = []
        if method == 'by_month':
            for i in range(MONTHS_PER_YEAR):
                train_i, test_i, val_i = sample_month(i, train_size, test_size, self.features_df.shape[0])
                train_indices = train_indices + train_i
                test_indices = test_indices + test_i
                val_indices = val_indices + val_i
        elif method == 'sequential':
            np.random.seed(22943)
            total_num_samples = self.features_df.as_matrix().shape[0]
            train_boundary = int(total_num_samples*train_size)
            val_boundary = int(total_num_samples*(train_size + test_size))
            # train_indices = np.random.choice(train_boundary, int(0.8*train_boundary), replace=False)
            train_indices = np.arange(0, train_boundary)
            val_indices = np.arange(train_boundary, val_boundary)
            test_indices = np.arange(val_boundary, total_num_samples)
        train_dfs = []
        test_dfs = []
        val_dfs = []
        for i in train_indices:
            train_dfs.append(ft.iloc[i].tolist())
        self.train_df = pd.DataFrame(train_dfs, columns=self.features_df.columns)
        for i in test_indices:
            test_dfs.append(ft.iloc[i].tolist())
        self.test_df = pd.DataFrame(test_dfs, columns=self.features_df.columns)
        for i in val_indices:
            val_dfs.append(ft.iloc[i].tolist())
        self.val_df = pd.DataFrame(val_dfs, columns=self.features_df.columns)
        self.train_df = self.data_scaler.scale_training_data(self.train_df, self.numerical_features)
        self.test_df = self.data_scaler.scale_testing_data(self.test_df, self.numerical_features)
        self.val_df = self.data_scaler.scale_testing_data(self.val_df, self.numerical_features)
        return self.train_df, self.val_df, self.test_df


    def convert_dfs_to_numpy(self,df):
        num_features = df.as_matrix().shape[1]
        return (df.ix[:, 0:num_features-1].as_matrix(), df.ix[:, num_features-1].as_matrix())

    def compute_Pearson_correlation(self):
        # Correlation analysis
        data_matrix = self.features_df[self.numerical_features].as_matrix()
        means = [np.mean(data_matrix[:, i]) for i in range(data_matrix.shape[1])]
        for i in range(len(means)):
            data_matrix[:, i] -= means[i]
        size = data_matrix.shape[0]
        covariance_matrix = np.dot(np.transpose(data_matrix), data_matrix)/(size - 1)
        stds = [np.sqrt(covariance_matrix[i, i]) for i in range(covariance_matrix.shape[0])]
        pearson_correlations = [covariance_matrix[i, -1]/(stds[i]*stds[-1]) for i in range(covariance_matrix.shape[0])]
        corr_dict = {}
        for idx,item in enumerate(fp.numerical_features):
            corr_dict[item] = pearson_correlations[idx]
        sorted_corr = sorted(corr_dict.items(), key=operator.itemgetter(1), reverse=True)
        for name, corr in sorted_corr:
            print('%s, %s' % (name, corr))

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

def sample_month(month_index, train_size, test_size, sample_size):
    np.random.seed(22943)
    indices = np.arange(0, HRS_PER_DAY*DAYS_PER_MONTH)
    set_indices = set(indices)
    train_indices = np.random.choice(indices,
                                  int(HRS_PER_DAY*DAYS_PER_MONTH*train_size),
                                  replace=False).tolist()
    test_indices = np.random.choice(list(set_indices.difference(set(train_indices))),
                                 int(HRS_PER_DAY*DAYS_PER_MONTH*test_size),
                                 replace=False).tolist()
    val_indices = list(set_indices.difference(set(train_indices)).difference(test_indices))
    train_i = []
    test_i = []
    val_i = []

    for i in train_indices:
        shifted_index = i + month_index*HRS_PER_DAY*DAYS_PER_MONTH
        if shifted_index > sample_size:
            shifted_index = sample_size - 1
        train_i += [shifted_index]
    for i in test_indices:
        shifted_index = i + month_index*HRS_PER_DAY*DAYS_PER_MONTH
        if shifted_index > sample_size:
            shifted_index = sample_size - 1
        test_i += [shifted_index]
    for i in val_indices:
        shifted_index = i + month_index*HRS_PER_DAY*DAYS_PER_MONTH
        if shifted_index > sample_size:
            shifted_index = sample_size - 1
        val_i += [shifted_index]
    return train_i, test_i, val_i

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]




if __name__ == '__main__':
    START_DATE = '2012-07-01'
    LOAD_ZONES = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']
    END_DATES = ['2013-12-31', '2014-12-31', '2015-12-31']
    fp = Feature_Processor()
    for lz in LOAD_ZONES:
        for ed in END_DATES:
            print(lz + ed)
            os.chdir('../normalized_datasets')
            f1 = open('%s_%s_seq_train_features.pkl' % (ed, lz), 'w+')
            f2 = open('%s_%s_seq_test_features.pkl' % (ed, lz), 'w+')
            f3 = open('%s_%s_seq_val_features.pkl' % (ed, lz), 'w+')
            f4 = open('%s_%s_seq_train_targets.pkl' % (ed, lz), 'w+')
            f5 = open('%s_%s_seq_test_targets.pkl' % (ed, lz), 'w+')
            f6 = open('%s_%s_seq_val_targets.pkl' % (ed, lz), 'w+')
            f7 = open('%s_%s_seq_scaler.pkl' % (ed, lz), 'w+')
            sfp.query(START_DATE, ed)
            sfp.construct_feature_vector_matrix(lz)
            train_features, train_targets, test_features, test_targets, val_features, val_targets = sfp.train_test_validate()
            pickle.dump(train_features, f1)
            pickle.dump(test_features, f2)
            pickle.dump(val_features, f3)
            pickle.dump(train_targets, f4)
            pickle.dump(test_targets, f5)
            pickle.dump(val_targets, f6)
            pickle.dump(sfp.sequence_scaler, f7)

