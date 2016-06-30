

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



#Acquire DAM SPP for the different settlementpoints for a specific date range
class Query_DAM_by_SP(Query_ERCOT_DB):
    settlement_points = []
        
    def __init__(self):
        Query_ERCOT_DB.c.execute("""SELECT DISTINCT settlement_point FROM DAM_prices_by_SPP""")
        r = list(Query_ERCOT_DB.c.fetchall())
        for settlement_point in r:
            if settlement_point[0] == "\"Settlement Point\"":
                continue
            self.settlement_points.append(settlement_point[0])
    
    #query for all prices for all load zones and hubs for specified date range
    def query(self,sd,ed):
        self.start_date = sd
        self.end_date = ed
        self.result_dict = {}
        for (idx,val) in enumerate(self.settlement_points):
            Query_ERCOT_DB.c.execute("""SELECT delivery_date,hour_ending,spp 
                FROM DAM_prices_by_SPP 
                WHERE settlement_point = "%s" 
                AND delivery_date > "%s" 
                AND delivery_date < "%s" 
                ORDER BY delivery_date,hour_ending""" % (val,self.start_date,self.end_date))
            self.result = list(Query_ERCOT_DB.c.fetchall())
            self.result_dict[val] = self.result
        self.spp_dict = {}
        self.dts = []
        self.count = 0
        for bus_name,result in self.result_dict.iteritems():
            self.spps = []
            for (date,time,spp) in self.result:
                time = str(int(time.split(":")[0])-1)
                dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
                self.spps.append(float(spp))
                if self.count == 0:
                    self.dts.append(dt)
            self.count = self.count + 1
            self.spp_dict[bus_name] = self.spps
        self.string_dts = [dt.strftime("%Y-%m-%d %H") for dt in self.dts]
        self.df = pd.DataFrame(data = self.spp_dict, index = self.string_dts)
        
    def construct_feature_vector_matrix(self,lzhub,model_type):
        self.dflzhub = self.df[lzhub]
        self.features = []
        if model_type == "A":
            for dt,price in self.dflzhub.iteritems():
                pred_hour_index = self.dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    self.features.append([work_day_or_holiday(string_to_date(dt)),                                           string_to_date(dt).hour,                                           string_to_date(dt).weekday()]                                          + self.dflzhub.iloc[pred_hour_index - 2*24:pred_hour_index - 1*24].tolist()                                          + self.dflzhub.iloc[pred_hour_index - 7*24:pred_hour_index - 6*24].tolist())
            self.feature_labels = ['Holiday','Hour','Day']                                + [('P(h-%s)' % str(i+1)) for i in range(24,48)][::-1]                                + [('P(h-%s)' % str(i+1)) for i in range(144,168)][::-1]
                
            self.idx_wout_1st_week = list(self.dflzhub.index.values)[7*24:]
            self.features_df = pd.DataFrame(data = self.features,                                            index = self.idx_wout_1st_week,                                            columns = self.feature_labels)
            self.features_df = encode_onehot(self.features_df,'Day')
            return self.features_df.join(self.dflzhub,how='left')
    
   
    
    def plot(self):
        #plot the data for a date range
        for bus_name,price in self.spp_dict.iteritems():
            if bus_name != "date_time":
                plt.plot(self.dts,price,label="%s" % bus_name)
        plt.title("SPP by LZ and HUB for %s" % start_date.split("-")[0])
        plt.xlabel("Date-Time")
        plt.ylabel("SPP")
        plt.legend()
        plt.show()

    def get_settlement_points(self):
        return self.settlement_points

def train_test_validate(ft):
    feature_target = ft.as_matrix()
    num_features = feature_target.shape[1]
    # split data into 24 hr samples and split into train, test, and validation sets
    samples = [feature_target[i*24:(i*24+24),:] for i in range(356)]
    np.random.seed(22943)
    indices = [i for i in range(356)]
    set_indices = set(indices)
    train_indices = np.random.choice(indices,int(356*0.5),replace=False)
    set_after_train_sample = set_indices.difference(set(train_indices))
    indices_after_train = list(set_after_train_sample)
    validate_indices = np.random.choice(indices_after_train,int(356*0.25),replace=False)
    test_indices = list(set_after_train_sample.difference(set(validate_indices)))
    train_list = [samples[i] for i in train_indices]
    validate_list = [samples[i] for i in validate_indices]
    test_list = [samples[i] for i in test_indices]
    train = train_list[0]
    validate = validate_list[0]
    test = test_list[0]
    for s in train_list[1:]:
        train = np.concatenate((train,s),axis=0)
    for s in validate_list[1:]:
        validate = np.concatenate((validate,s),axis=0)
    for s in test_list[1:]:
        test = np.concatenate((test,s),axis=0)
    train = (train[:,:num_features-1],train[:,num_features-1])
    validate = (validate[:,:num_features-1],validate[:,num_features-1])
    test = (test[:,:num_features-1],test[:,num_features-1])
    return (train,validate,test)

def string_to_date(string_date):
    return datetime.strptime(string_date,"%Y-%m-%d %H")

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
    data = enc.fit_transform(df[cols].reshape(-1,1)).toarray()
    one_hot_df = pd.DataFrame(data = data,                              index = index,                              columns = [cols + '%s' % i for i in range(data.shape[1])])
    del df[cols]
    return df.join(one_hot_df,how='inner')



#qdsp = Query_DAM_by_SP()
#qdsp.query("2012-01-01","2012-12-31")
#qdsp.plot()
#feature_targets = qdsp.construct_feature_vector_matrix("HB_BUSAVG","A")



#train, val, test = train_test_validate(feature_targets)
#train[0]

