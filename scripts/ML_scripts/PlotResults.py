__author__ = 'kenlee'

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

zones = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']
colors = ['r', 'b', 'g', 'c', 'w']
df = pd.read_csv('/home/kenlee/energy_market_project/scripts/test_results/MLP_ModelA_results.csv')
index = np.arange(5)
bar_width = 0.1
i = 0
for z in zones:
    plt.bar(index + i*bar_width,
            df.loc[df['zone']==z]['TheilU1'],
            bar_width,
            color=colors[i],
            label=z)
    i = i + 1
plt.xlabel('Year')
plt.ylabel('Theil U1 Statistic')
plt.xticks(index + bar_width, ('2011', '2012', '2013', '2014', '2015'))
plt.legend()
plt.tight_layout()
plt.show()


