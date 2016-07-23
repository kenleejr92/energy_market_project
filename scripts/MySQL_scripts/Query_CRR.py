__author__ = 'kenlee'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Query_ERCOT_DB import Query_ERCOT_DB

class Query_CRR(Query_ERCOT_DB):

    def __init__(self):
        self.crr_periods = None
        self.sources = {}
        self.sinks = {}
        self.graph_list = []
        Query_ERCOT_DB.c.execute("""SELECT StartDate, EndDate
            FROM crr_ownership GROUP BY StartDate, EndDate""")
        self.crr_periods = list(Query_ERCOT_DB.c.fetchall())[1:]
        Query_ERCOT_DB.c.execute("""SELECT DISTINCT Source
            FROM crr_ownership ORDER BY Source""")
        sources = list(Query_ERCOT_DB.c.fetchall())
        sources = [r[0] for r in sources]
        for idx, src in enumerate(sources):
            self.sources[src] = idx
        Query_ERCOT_DB.c.execute("""SELECT DISTINCT Sink
            FROM crr_ownership ORDER BY Sink""")
        sinks = list(Query_ERCOT_DB.c.fetchall())
        sinks = [r[0] for r in sinks]
        for idx, snk in enumerate(sinks):
            self.sinks[snk] = idx
        self.capacity_matrix = np.zeros((len(sources), len(sources)))


    def query(self,option='all',start_date='2012-12-01',end_date='2012-12-31'):
        if option == 'all':
            for sd, ed in crr_periods:
                Query_ERCOT_DB.c.execute("""select Source, Sink, sum(MW)
                FROM crr_ownership
                WHERE StartDate >= "%s"
                AND EndDate <= "%s"
                GROUP BY Source, Sink""" % (sd,ed))
            self.graph_list.append(list(Query_ERCOT_DB.c.fetchall()))
        elif option == 'date_range':
            Query_ERCOT_DB.c.execute("""select Source, Sink, sum(MW)
                FROM crr_ownership
                WHERE StartDate >= "%s"
                AND EndDate <= "%s"
                GROUP BY Source, Sink""" % (start_date, end_date))
            temp = list(Query_ERCOT_DB.c.fetchall())
            for source, sink, MW_total in temp:
                self.capacity_matrix[qcrr.sources[source], qcrr.sinks[sink]] = MW_total



qcrr = Query_CRR()
crr_periods = qcrr.query(option = 'date_range')
for row in qcrr.capacity_matrix:
    print(row)