
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib qt')
import time
import sys
import math
import os
import numpy as np
import pandas as pd
import MySQLdb
from datetime import datetime
import calendar
import holidays
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/kenleejr92/energy_market_project/scripts/MySQL_scripts')
from Query_ERCOT_DB import Query_ERCOT_DB
import cPickle as pickle
from datetime import datetime
#import findspark
#findspark.init()
#from pyspark import SparkConf, SparkContext
#from pyspark.mllib.linalg import Matrix, Matrices
#import pyspark.mllib.linalg.distributed as pydist
#from pyspark.sql import SQLContext
#from pyspark.mllib.stat import Statistics
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
#import mpld3
import re
from sets import Set
#mpld3.enable_notebook()

class LMP_Query(Query_ERCOT_DB):

    def __init__(self):
        self.SPPs = ['HB_BUSAVG', 
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
        self.node_dict = {}
        self.table_list = []
        self.df = None
        self.CRR_nodes = None
        self.table_boundaries = {'table0':('0001', 'BLUEMD1_8X'),
                                 'table1':('BLUEMD1_8Z', 'CHT_M'),
                                 'table2':('CHT_X', 'DUKE_8405'),
                                 'table3':('DUKE_8505', 'ELEVEE_E8'),
                                 'table4':('ELEVEE_W8', 'GREENLK_L_A'),
                                 'table5':('GREENLK_L_B', 'KEETER'),
                                 'table6':('KEITH', 'L_CEDAHI8_1Y'),
                                 'table7':('L_CEDAHI8_1Z', 'MOSES_1G'),
                                 'table8':('MOSES_2G', 'PHR_8135'),
                                 'table9':('PHR_8140', 'SANDOW1_8Y'),
                                 'table10':('SANDOW_4G', 'TCN7225_BUS'),
                                 'table11': ('TCN7230_BUS', 'VENSW_1777'),
                                 'table12':('VENSW_1785', '_WC_V_C')
                                 }
        for i in range(0,13):
            Query_ERCOT_DB.c.execute("""SHOW columns FROM DAM_LMP%s""" % i)
            result = [r[0] for r in Query_ERCOT_DB.c.fetchall()[2:]]
            self.table_list.append(result)
            for node in result:
                self.node_dict[node] = i
        
    
    def get_CRR_nodes(self):
        Query_ERCOT_DB.c.execute("""SELECT DISTINCT Sink FROM crr_ownership ORDER BY Sink""")
        nodes = list(Query_ERCOT_DB.c.fetchall())
        CRR_prefixes = [r[0] for r in nodes]
        pattern = re.compile('.*_')
        matching_patterns = Set()
        for idx, node in enumerate(CRR_prefixes):
            matches = re.findall(pattern, node)
            if matches: matching_patterns.add(matches[0][:-1])
        # flatten list of lists      
        all_nodes = [item for sublist in self.table_list for item in sublist] + self.SPPs
        self.CRR_nodes = []
        for pattern in matching_patterns:
            pattern2 = re.compile('(.*%s.*)' % pattern)
            for node in all_nodes:
                if re.search(pattern2, node):
                    self.CRR_nodes.append(node)
        return self.CRR_nodes
    
    def query_single_node(self, node):
        s="""SELECT delivery_date, hour_ending, %s from DAM_LMP%s order by delivery_date, hour_ending""" % (node, self.node_dict[node])
        Query_ERCOT_DB.c.execute(s)
        result = list(Query_ERCOT_DB.c.fetchall())
        fresult = []
        for r in result:
            temp = ()
            date = r[0]
            time = str(int(r[1].split(":")[0])-1)
            dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
            for x in r[2:]:
                if x == None: x = 0
                temp = temp + (float(x),)
            r = (dt,) + temp
            fresult.append(r)
        self.df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=[node])
        return self.df
        
    def query(self, nodes, start_date='2011-01-01', end_date='2015-5-23'):
        node_string=''
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
        Query_ERCOT_DB.c.execute(s)
        result = list(Query_ERCOT_DB.c.fetchall())
        fresult = []
        for r in result:
            temp = ()
            date = r[0]
            time = str(int(r[1].split(":")[0])-1)
            dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
            for x in r[2:]:
                if x == None: x = 0
                temp = temp + (float(x),)
            r = (dt,) + temp
            fresult.append(r)
        self.df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=nodes)
        return self.df
        
    def get_price(self, node, date, hour_ending):
        for i in range(0,13):
            node = append_n(node)
            if node in self.table_columns['table%s' % i]:
                Query_ERCOT_DB.c.execute("""SELECT %s FROM DAM_LMP%s WHERE delivery_date = "%s" AND hour_ending = \"%s\"""" % (node, i, date, hour_ending))
                result = list(Query_ERCOT_DB.c.fetchall())[0][0]
                return result
        

def append_n(name):
    if name[0] in ['0','1','2','3','4','5','6','7','8','9'] or name == 'LOAD':
        name = 'n' + name
    return name

def dist(x,y):
    cost = 0
    for i in range(len(x)):
        if x[i] == 0 or y[i] == 0:
            continue
        else:
            cost = cost + np.abs(y[i] - x[i])[0]
    return cost

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


