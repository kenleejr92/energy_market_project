__author__ = 'kenlee'

import numpy as np
import pandas as pd
from pandas import Series

df = pd.read_csv('/home/kenlee/Energy_Market_Project/LMP_data/cdr.00012328.0000000000000000.20160419.122930.DAMHRLMPNP4183.csv')
df = df.sort_values(by=['BusName'])
del df['DSTFlag']
nodes = df.BusName.unique()
init_df = df.loc[df['BusName'] == '0001']
# for node in nodes:
#     temp_df = df.loc[df['BusName'] == node][['LMP']].rename(columns = {'LMP': node})


