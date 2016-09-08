__author__ = 'kenlee'

from Query_ERCOT_DB import Query_ERCOT_DB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.signal
import math
from sklearn import preprocessing

class Adaptive_Normalizer(Query_ERCOT_DB):

    def __init__(self):
        self.zone = None
        self.start_date = None
        self.end_date = None
        self.df = None
        self.price_series = None
        self.MA = None

    def query(self, zone, sd, ed):
        self.zone = zone
        self.start_date = sd
        self.end_date = ed
        Query_ERCOT_DB.c.execute("""SELECT delivery_date, hour_ending, %s FROM DAM_SPPs
                WHERE DAM_SPPs.delivery_date > "%s"
                AND DAM_SPPs.delivery_date < "%s"
                ORDER BY DAM_SPPs.delivery_date, DAM_SPPs.hour_ending""" % (zone, sd, ed))
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
        self.df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=[zone])
        self.price_series = self.df[zone].as_matrix()

    def plot(self, x):
        plt.plot(x)
        plt.show()

    def moving_average(self, k):
        x = self.price_series
        self.MA = scipy.signal.lfilter(np.ones(k)/k, [1], x)[k:]
        return self.MA

    def create_sliding_window(self,w):
        R = np.zeros(len(self.price_series)+1)
        for i, item in enumerate(self.price_series):
            R[i+1] = self.price_series[math.ceil((i+1)/w)*i % w]/self.MA[math.ceil((i+1)/w)]
        return R

if __name__ == '__main__':
    AN = Adaptive_Normalizer()
    AN.query('LZ_WEST', '2011-01-01', '2015-12-31')

    # plt.plot(AN.price_series)
    # plt.plot((AN.moving_average(10)))
    AN.moving_average(10)
    R = AN.create_sliding_window(72)
    mean = np.mean(R)
    std = np.std(R)
    train = AN.price_series[0:int(0.6*len(AN.price_series))]
    test = AN.price_series[int(0.6*len(AN.price_series)):]
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler2 = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x = scaler.fit_transform(train.reshape(-1,1))

    y = scaler.transform(test.reshape(-1,1))
    y2 = scaler2.fit_transform(test.reshape(-1,1))
    plt.plot(test)
    plt.plot(scaler.inverse_transform(y2))
    # plt.plot(x)
    # plt.plot(AN.price_series)
    plt.show()