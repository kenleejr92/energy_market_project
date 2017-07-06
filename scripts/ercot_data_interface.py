
import time
import sys
import math
import os
import numpy as np
import pandas as pd
import pymysql
from datetime import datetime
import re
import calendar
import matplotlib.pyplot as plt
from sets import Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler

HOST = 'localhost'
USER = 'root'
PASSWORD = 'yangspike92'
DB = 'ercot_data'
SPPs = ['HB_BUSAVG', 
        'HB_HOUSTON', 
        'HB_HUBAVG', 
        'HB_NORTH', 
        'HB_SOUTH', 
        'HB_WEST', 
        'LZ_AEN', 
        'LZ_CPS', 
        'LZ_HOUSTON', 
        'LZ_LCRA', 
        'LZ_NORTH', 
        'LZ_RAYBN', 
        'LZ_SOUTH',  
        'LZ_WEST']   

def weekday_of_date(date):
    return calendar.day_name[date.weekday()]

#def work_day_or_holiday(date):
#    us_holidays = holidays.UnitedStates()
#    if date in us_holidays or weekday_of_date(date) == "Sunday" or weekday_of_date(date) == "Saturday":
#        return int(1)
#    else: return int(0)

class ercot_data_interface(object):

    def __init__(self, password=PASSWORD):
        self.train_fraction = 0.8
        self.connection = pymysql.connect(host=HOST, user=USER, password=password, db=DB, port=3306,  cursorclass=pymysql.cursors.Cursor)
        self.all_nodes = []
        self.all_nodes_dict = {}
        self.standard_scaler = None
        with self.connection.cursor() as cursor:                      
            for i in range(0,13):
                cursor.execute("""SHOW columns FROM DAM_LMP%s""" % i)
                result = cursor.fetchall()
                result = [r[0] for r in list(result[2:])]
                self.all_nodes = self.all_nodes + result
                for node in self.all_nodes:
                    self.all_nodes_dict[node] = i
        self.all_nodes = self.all_nodes + SPPs

    def query_prices(self, nodes, start_date, end_date):
        node_string=''
        if isinstance(nodes, str): 
            node_string = node_string + nodes
        else:         
            for node in nodes:
                node_string = node_string + node + ',' + ' '
            node_string = node_string[:-2]
        s = """SELECT DAM_LMP0.delivery_date, DAM_LMP0.hour_ending, %s 
                                    from DAM_LMP0 join DAM_LMP1 on (DAM_LMP0.delivery_date = DAM_LMP1.delivery_date and DAM_LMP0.hour_ending = DAM_LMP1.hour_ending) 
                                    join DAM_LMP2 on (DAM_LMP0.delivery_date = DAM_LMP2.delivery_date and DAM_LMP0.hour_ending = DAM_LMP2.hour_ending)
                                    join DAM_LMP3 on (DAM_LMP0.delivery_date = DAM_LMP3.delivery_date and DAM_LMP0.hour_ending = DAM_LMP3.hour_ending)
                                    join DAM_LMP4 on (DAM_LMP0.delivery_date = DAM_LMP4.delivery_date and DAM_LMP0.hour_ending = DAM_LMP4.hour_ending)
                                    join DAM_LMP5 on (DAM_LMP0.delivery_date = DAM_LMP5.delivery_date and DAM_LMP0.hour_ending = DAM_LMP5.hour_ending)
                                    join DAM_LMP6 on (DAM_LMP0.delivery_date = DAM_LMP6.delivery_date and DAM_LMP0.hour_ending = DAM_LMP6.hour_ending)
                                    join DAM_LMP7 on (DAM_LMP0.delivery_date = DAM_LMP7.delivery_date and DAM_LMP0.hour_ending = DAM_LMP7.hour_ending)
                                    join DAM_LMP8 on (DAM_LMP0.delivery_date = DAM_LMP8.delivery_date and DAM_LMP0.hour_ending = DAM_LMP8.hour_ending)
                                    join DAM_LMP9 on (DAM_LMP0.delivery_date = DAM_LMP9.delivery_date and DAM_LMP0.hour_ending = DAM_LMP9.hour_ending)
                                    join DAM_LMP10 on (DAM_LMP0.delivery_date = DAM_LMP10.delivery_date and DAM_LMP0.hour_ending = DAM_LMP10.hour_ending)
                                    join DAM_LMP11 on (DAM_LMP0.delivery_date = DAM_LMP11.delivery_date and DAM_LMP0.hour_ending = DAM_LMP11.hour_ending)
                                    join DAM_LMP12 on (DAM_LMP0.delivery_date = DAM_LMP12.delivery_date and DAM_LMP0.hour_ending = DAM_LMP12.hour_ending)
                                    join DAM_SPPs on (DAM_LMP0.delivery_date = DAM_SPPs.delivery_date and DAM_LMP0.hour_ending = DAM_SPPs.hour_ending)
                                    where DAM_LMP0.delivery_date > "%s" and DAM_LMP0.delivery_date < "%s" order by DAM_LMP0.delivery_date, DAM_LMP0.hour_ending;""" % (node_string, start_date, end_date)
                
        with self.connection.cursor() as cursor:            
            # Create a new record            
            cursor.execute(s)
            list_data = []
            tuple_data = cursor.fetchall()
            for t in tuple_data:
                row = []
                date = t[0].strftime("%Y-%m-%d")
                time = str(int(t[1].seconds/3600))
                dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
                for price in t[2:]:
                    if price == None: price = 0.
                    row.append(float(price))
                r = [dt] + row
                list_data.append(r)
            df = pd.DataFrame(data=[l[1:] for l in list_data], index=[l[0] for l in list_data])  
            return df.sort_index()

    def get_sources_sinks(self):
        with self.connection.cursor() as cursor:                      
            cursor.execute("""SELECT DISTINCT Sink FROM crr_ownership ORDER BY Sink""")
            result = cursor.fetchall()
            CRR_prefixes1 = Set([r[0] for r in result])
            cursor.execute("""SELECT DISTINCT Source FROM crr_ownership ORDER BY Source""")
            result = cursor.fetchall()
            CRR_prefixes2 = Set([r[0] for r in result])
            CRR_prefixes = CRR_prefixes1.intersection(CRR_prefixes2)
            return list(CRR_prefixes)

    

    def get_nearest_CRR_neighbors(self, node):   
        nn = []    
        with self.connection.cursor() as cursor:            
            # Create a new record            
            cursor.execute("""SELECT DISTINCT Source FROM crr_ownership WHERE Source = \"%s\"""" % node)
            result = cursor.fetchall()
            sources = Set([r[0] for r in result])
            cursor.execute("""SELECT DISTINCT Sink FROM crr_ownership WHERE Sink = \"%s\"""" % node)
            result = cursor.fetchall()
            sinks = Set([r[0] for r in result])
            nearest_neighbors = sinks.intersection(sources)

            pattern = re.compile('.*_')
            matching_patterns = Set()
            for idx, node in enumerate(nearest_neighbors):
                matches = re.findall(pattern, node)
                if matches: matching_patterns.add(matches[0][:-1])  
            for pattern in matching_patterns:
                pattern2 = re.compile('(.*%s.*)' % pattern)
                for nodex in self.all_nodes:
                    if re.search(pattern2, nodex):
                        nn.append(nodex)
            for i, n in enumerate(nn):
                if n not in self.all_nodes:
                    del nn[i]
            if node not in self.all_nodes:
                return None
            else:
                return [node] + nn

    def get_CRR_nodes(self):
        all_nodes = []
        all_nodes_dict = {}
        with self.connection.cursor() as cursor:                      
            cursor.execute("""SELECT DISTINCT Sink FROM crr_ownership ORDER BY Sink""")
            result = cursor.fetchall()
            CRR_prefixes1 = Set([r[0] for r in result])
            cursor.execute("""SELECT DISTINCT Source FROM crr_ownership ORDER BY Source""")
            result = cursor.fetchall()
            CRR_prefixes2 = Set([r[0] for r in result])
            CRR_prefixes = CRR_prefixes1.intersection(CRR_prefixes2)
            pattern = re.compile('.*_')
            matching_patterns = Set()
            for idx, node in enumerate(CRR_prefixes):
                matches = re.findall(pattern, node)
                if matches: matching_patterns.add(matches[0][:-1])   
            CRR_nodes = []
            for pattern in matching_patterns:
                pattern2 = re.compile('(.*%s.*)' % pattern)
                for node in self.all_nodes:
                    if re.search(pattern2, node):
                        CRR_nodes.append(node)
            return CRR_nodes


    def get_train_test(self, node, normalize=True, include_seasonal_vectors=True, log_difference=False):
        X = self.query_prices(node, '2011-01-01', '2016-5-23')
        datetimes = X.index
        X = X.as_matrix()
        if include_seasonal_vectors:
            hours = []
            months = []
            weekdays = []
            for dt in datetimes:
                dt = dt.to_pydatetime()
                weekday = calendar.weekday(dt.year, dt.month, dt.day)
                hours.append(dt.hour)
                months.append(dt.month)
                weekdays.append(weekday)
            hours = np.array(hours).reshape(-1, 1)
            months = np.array(months).reshape(-1, 1)
            weekdays = np.array(weekdays).reshape(-1, 1)
            X = np.hstack((X, hours, months, weekdays))
        train_stop = int(self.train_fraction*X.shape[0])
        train = X[:train_stop]
        test = X[train_stop:]
        if normalize == True:
                self.standard_scaler = StandardScaler()
                train = self.standard_scaler.fit_transform(train)
                test = self.standard_scaler.transform(test)
        if log_difference == True:
            train = np.log(train[1:]) - np.log(train[:-1])
            test = np.log(test[1:]) - np.log(test[:-1])
        return train, test



if __name__ == '__main__':
    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    #source_sinks[20] gave an error
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[100])
    if nn == None:
        print 'Prices not found'
    else:
        train, test = ercot.get_train_test(nn[0], include_seasonal_vectors=False)
        plt.plot(train)
        plt.show()
        # scaler = MinMaxScaler((-1,1))
        # train = scaler.fit_transform(train)
        # mu = 1024
        # F = np.sign(train)*np.log(1 + mu*np.abs(train))/np.log(1 + mu)
        # F = ((F + 1) / 2 * mu + 0.5).astype('int')
        # print F
        # plt.plot(F)
        # plt.show()
    # plt.plot(train)
    # plt.show()
    # print train.shape
    # plt.plot(train, label='train')
    # plt.plot(test, label='test')
    # plt.plot(val, label='val')
    # plt.legend()
    # plt.show()


