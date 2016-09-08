__author__ = 'kenlee'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from Query_ERCOT_DB import Query_ERCOT_DB

class Query_CRR(Query_ERCOT_DB):

    def __init__(self):
        self.table_columns = {}
        self.df = None
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
        self.table_dict = {'table0':[],
                           'table1':[],
                           'table2':[],
                           'table3':[],
                           'table4':[],
                           'table5':[],
                           'table6':[],
                           'table7':[],
                           'table8':[],
                           'table9':[],
                           'table10':[],
                           'table11':[],
                           'table12':[]}
        for i in range(0,13):
            Query_ERCOT_DB.c.execute("""SHOW columns FROM DAM_LMP%s""" % i)
            result = [r[0] for r in Query_ERCOT_DB.c.fetchall()[2:]]
            self.table_columns['table%s' % i] = result

    def query(self, node, start_date, end_date):
        Query_ERCOT_DB.c.execute("""SELECT Source FROM crr_ownership WHERE Sink = '%s' GROUP BY Source""" % node)
        nearest_sources = [r[0] for r in list(Query_ERCOT_DB.c.fetchall())]
        Query_ERCOT_DB.c.execute("""SELECT Sink FROM crr_ownership WHERE Source = '%s' GROUP BY Sink""" % node)
        nearest_sinks = [r[0] for r in list(Query_ERCOT_DB.c.fetchall())]
        nearest_neighbors = list(set(nearest_sources).union(set(nearest_sinks)))
        for nn in nearest_neighbors:
            for i in range(0,13):
                nn = append_n(nn)
                if nn in self.table_columns['table%s' % i]:
                    self.table_dict['table%s' % i].append(nn)

        i=0
        for table, columns in self.table_dict.iteritems():
            if len(columns) == 0:
                continue
            column_string=''
            table_num = table[5:]
            for c in columns:
                column_string = column_string + c + ','
            column_string = column_string[:-1]
            Query_ERCOT_DB.c.execute("""SELECT delivery_date, hour_ending,%s FROM DAM_LMP%s WHERE delivery_date > "%s" AND delivery_date < "%s" ORDER BY delivery_date, hour_ending""" % (column_string, table_num, start_date, end_date))
            result = list(Query_ERCOT_DB.c.fetchall())
            fresult = []
            for r in result:
                temp = ()
                date = r[0]
                time = str(int(r[1].split(":")[0])-1)
                dt = datetime.strptime(date + " " + time, "%Y-%m-%d %H")
                for x in r[2:]:
                    if x == None:
                        temp = temp + (0,)
                    else: temp = temp + (float(x),)
                r = (dt,) + temp
                fresult.append(r)
            if i == 0:
                self.df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=columns)
            else:
                df = pd.DataFrame(data=[f[1:] for f in fresult], index=[r[0] for r in fresult], columns=columns)
                self.df = self.df.join(df, how='left')
            i = i + 1

    def plot(self):
        self.df.plot()
        plt.title("DAM LMP by for CRR nodes")
        plt.xlabel("Date-Time")
        plt.ylabel("LMP")
        legend = plt.legend()
        legend.remove()
        plt.show()

def append_n(name):
    if name[0] in ['0','1','2','3','4','5','6','7','8','9'] or name == 'LOAD':
        name = 'n' + name
    return name

if __name__ == '__main__':
    qcrr = Query_CRR()
    qcrr.query('LZ_WEST','2012-01-01','2012-12-31')
    print(qcrr.df)