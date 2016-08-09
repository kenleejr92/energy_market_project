__author__ = 'kenlee'

from DAM_prices_by_SP import Feature_Processor
from Query_ERCOT_DB import Query_ERCOT_DB
import pandas as pd
import numpy as np
from datetime import datetime

class Correlation_Analyzer(Feature_Processor):

    def __init__(self):
        self.include = None
        self.table_headers = []
        self.features_df = None
        self.data_matrix = None
        Query_ERCOT_DB.c.execute("""SHOW COLUMNS FROM DAM_SPPs""")
        r = list(Query_ERCOT_DB.c.fetchall())
        for sp in r:
            if sp[0] == "delivery_date" or sp[0] == "hour_ending":
                continue
            self.table_headers.append(sp[0])

    def query(self, sd, ed, include='price_only'):
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

    def construct_feature_vector_matrix(self, lzhub, model_type='A'):
        self.lzhub = lzhub + '_SPP'
        dflzhub = self.df[lzhub + '_SPP']
        features = []
        for dt, price in dflzhub.iteritems():
                pred_hour_index = dflzhub.index.get_loc(dt)
                if pred_hour_index - 7*24 >= 0:
                    feature_row = [dflzhub.iloc[pred_hour_index - i] for i in range(24, 169)]
                    features.append(feature_row)
        feature_labels = ['P(h-%s)' % i for i in range(24, 169)]
        self.numerical_features = ['P(h-%s)' % i for i in range(24, 169)] + [lzhub + '_SPP']
        idx_wout_1st_week = list(dflzhub.index.values)[7*24:]

        self.features_df = pd.DataFrame(data=features,
                                   index=idx_wout_1st_week,
                                   columns=feature_labels)
        self.features_df = self.features_df.join(dflzhub, how='left')

        return self.features_df

    def compute_Pearson_correlation(self):
        pass


if __name__ == '__main__':
    ca = Correlation_Analyzer()
    ca.query('2011-01-01', '2011-12-31')
    ca.construct_feature_vector_matrix('LZ_WEST')
    print(ca.features_df)