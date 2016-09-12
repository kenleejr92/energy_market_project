__author__ = 'kenlee'
from Query_ERCOT_DB import Query_ERCOT_DB
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from Adaptive_Normalizer import reject_outliers, Adaptive_Normalizer
from Data_Scaler import Data_Scaler
from sklearn import preprocessing

class Exogeneous_Variables(Query_ERCOT_DB):

    def __init__(self):
        self.wind_df = None
        self.load_df = None
        self.load_forecast_df = None
        self.start_date = None
        self.end_date = None
        self.wind_headers = ['WGRPP_HOUSTON', 'WGRPP_NORTH', 'WGRPP_SOUTH', 'WGRPP_WEST']
        self.load_header = ['SystemTotal']

    def query_wind(self, sd, ed):
        self.start_date = sd
        self.end_date = ed
        Query_ERCOT_DB.c.execute("""SELECT delivery_date, WGRPP_HOUSTON, WGRPP_NORTH, WGRPP_SOUTH, WGRPP_WEST FROM Wind_forecasts
                WHERE delivery_date > "%s"
                AND delivery_date < "%s"
                ORDER BY delivery_date""" % (sd, ed))
        result = list(Query_ERCOT_DB.c.fetchall())
        fresult = []
        for r in result:
            temp = ()
            dt = datetime.strptime(r[0], '%Y-%m-%d %H:%M:%S')
            for x in r[1:]:
                if x == None:
                    temp = temp + (None,)
                else: temp = temp + (float(x),)
            r = (dt,) + temp
            fresult.append(r)
        self.wind_df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=self.wind_headers)
        return self.wind_df

    def query_load(self, sd, ed):
        self.start_date = sd
        self.end_date = ed
        Query_ERCOT_DB.c.execute("""SELECT delivery_date, hour_ending, SystemTotal FROM Load_forecast_by_WZ
                WHERE delivery_date > "%s"
                AND delivery_date < "%s"
                ORDER BY delivery_date, hour_ending""" % (sd, ed))
        result = list(Query_ERCOT_DB.c.fetchall())
        fresult = []
        for r in result:
            temp = ()
            date = r[0]
            time = r[1]
            d = datetime.strptime(date, '%Y-%m-%d')
            if d >= datetime.strptime('2012-06-27', '%Y-%m-%d'):
                hour = int(time.split(':')[0])
                if hour != 00:
                    time = str(int(time.split(':')[0])-1) + ':00:00'
                dt = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
            else: dt = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
            for x in r[2:]:
                if x == '0.0000':
                    temp = temp + (None,)
                else: temp = temp + (float(x),)
            r = (dt,) + temp
            fresult.append(r)
        self.load_df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=self.load_header)
        return self.load_df

if __name__ == '__main__':
    EV = Exogeneous_Variables()
    EV.query_wind('2012-01-01', '2015-12-31')
    EV.query_load('2012-01-01', '2015-12-31')
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    wind_load_df = EV.wind_df.join(EV.load_df, how='inner')
    wind_load_df[['SystemTotal', 'WGRPP_WEST']] = imp.fit_transform(wind_load_df[['SystemTotal', 'WGRPP_WEST']])
    wind_load_df = wind_load_df[['SystemTotal', 'WGRPP_WEST']]
    wind = wind_load_df['WGRPP_WEST'].as_matrix()
    load = wind_load_df['SystemTotal'].as_matrix()
    scaler_wind = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler_load = preprocessing.MinMaxScaler(feature_range=(0,1))
    wind = scaler_wind.fit_transform(wind.reshape(-1,1))
    load = scaler_load.fit_transform(load.reshape(-1,1))
    # wind = reject_outliers(wind,3)
    # load = reject_outliers(load,3)
    AN = Adaptive_Normalizer()
    AN.query('LZ_WEST', '2012-01-01', '2015-12-31')
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(AN.price_series.reshape(-1,1))
    # x = reject_outliers(x, 4)
    plt.plot(wind)
    plt.plot(load)
    plt.plot(x)
    plt.show()
