__author__ = 'kenlee'

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

zones = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']
df = pd.read_csv('/home/kenlee/energy_market_project/scripts/test_results/Model_results.csv')
print(df)
# index = np.arange(5)
# bar_width = 0.1
# i = 0
# f = plt.figure(1)
# for z in zones:
#     plt.bar(index + i*bar_width,
#             df.loc[df['zone']==z]['TheilU1'],
#             bar_width,
#             alpha=0.4,
#             color='r',
#             label=z)
#     plt.bar(index + i*bar_width,
#             dfB.loc[dfB['zone']==z]['TheilU1'],
#             bar_width,
#             alpha=0.8,
#             color='b',
#             label=z)
#     plt.bar(index + i*bar_width,
#             dfC.loc[dfB['zone']==z]['TheilU1'],
#             bar_width,
#             alpha=0.8,
#             color='g',
#             label=z)
#     i = i + 1
# plt.xlabel('Year')
# plt.ylabel('Theil U1 Statistic')
# plt.xticks(index + bar_width, ('2011', '2012', '2013', '2014', '2015'))
# # plt.legend()
# plt.tight_layout()
#
# i=0
# g = plt.figure(2)
# for z in zones:
#     plt.bar(index + i*bar_width,
#             dfA.loc[dfA['zone']==z]['MAPE'],
#             bar_width,
#             alpha=0.4,
#             color='r',
#             label=z)
#     plt.bar(index + i*bar_width,
#             dfB.loc[dfB['zone']==z]['MAPE'],
#             bar_width,
#             alpha=0.8,
#             color='b',
#             label=z)
#     plt.bar(index + i*bar_width,
#             dfC.loc[dfB['zone']==z]['MAPE'],
#             bar_width,
#             alpha=0.8,
#             color='g',
#             label=z)
#     i = i + 1
# plt.xlabel('Year')
# plt.ylabel('MAPE')
# plt.xticks(index + bar_width, ('2011', '2012', '2013', '2014', '2015'))
# plt.legend()
# plt.tight_layout()
# plt.show()
