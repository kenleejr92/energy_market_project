
# coding: utf-8

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
sys.path.insert(0, '/home/kenlee/energy_market_project/scripts/MySQL_scripts/')
from DAM_prices_by_SP import Feature_Processor
import time


lzhub = 'LZ_NORTH'
model_type = 'A'
fp = Feature_Processor()
fp.query('2012-01-01', '2012-12-31')
feature_df = fp.construct_feature_vector_matrix(lzhub, model_type)
fp.train_test_validate()
norm_train_set = fp.scale_num_features(lzhub, model_type, fp.train_df, 'standard_scale', set_type = 'Train')
norm_test_set = fp.scale_num_features(lzhub, model_type, fp.test_df, 'standard_scale', set_type = 'Test')
print(norm_train_set)
final_train_set = fp.convert_dfs_to_numpy(norm_train_set)
final_test_set = fp.convert_dfs_to_numpy(norm_test_set)
x_train, y_train = final_train_set
x_test, y_test = final_test_set


model = Sequential()
model.add(Dense(30, init = 'glorot_uniform', activation = 'tanh', input_dim = x_train.shape[1]))
model.add(Dense(1, init = 'zero', activation = 'linear'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, nb_epoch=100)

rescaled_test = fp.inverse_standard_scale(lzhub, model_type, norm_test_set)
rescaled_test = rescaled_test[:, -1]
plt.plot(rescaled_test, label='actual')
y_pred = model.predict(x_test)
norm_test_set[lzhub + '_SPP'] = y_pred
scaled_pred = fp.inverse_standard_scale(lzhub, model_type, norm_test_set)
scaled_pred = scaled_pred[:, -1]
plt.plot(scaled_pred, label='predicted')
plt.legend()
plt.show()

MAPE = np.sum(np.divide(np.abs(rescaled_test - scaled_pred), rescaled_test)*100)/rescaled_test.shape[0]
print(MAPE)





