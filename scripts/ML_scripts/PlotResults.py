__author__ = 'kenlee'

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

zones = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']
years = [2011, 2012, 2013, 2014, 2015]
df = pd.read_csv('/home/kenlee/energy_market_project/scripts/test_results/Model_results.csv')
def plot_results(statistic='MAPE'):
    bar_width = 0.1
    index = np.arange(len(zones))
    plt.figure()
    for j in range(len(zones)):


        plt.bar(index[j] + j*bar_width,
            np.mean(df.loc[(df['Zone']==zones[j]) & (df['Model']=='StackedLSTM')][statistic]),
            bar_width,
            alpha=0.8,
            color='g')
        plt.bar(index[j] + j*bar_width,
            np.mean(df.loc[(df['Zone']==zones[j]) & (df['Model']=='LSTM')][statistic]),
            bar_width,
            alpha=0.8,
            color='b')
        plt.bar(index[j] + j*bar_width,
            np.mean(df.loc[(df['Zone']==zones[j])  & (df['Model']=='SimpleRNN')][statistic]),
            bar_width,
            alpha=0.8,
            color='r')
        plt.bar(index[j] + j*bar_width,
            np.mean(df.loc[(df['Zone']==zones[j]) & (df['Model']=='MLPA')][statistic]),
            bar_width,
            alpha=0.8,
            color='y')

    plt.xticks(index + bar_width, tuple(zones))
    yellow_patch = mpatches.Patch(color='y', label='MLP')
    red_patch = mpatches.Patch(color='r', label='SimpleRNN')
    blue_patch = mpatches.Patch(color='b', label='LSTM')
    green_patch = mpatches.Patch(color='g', label='Stacked LSTM')
    plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])
    plt.xlabel('Load Zones')
    plt.ylabel(statistic)
    plt.show()


plot_results('MAPE')
plot_results('TheilU1')
plot_results('TheilU2')