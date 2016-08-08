
# coding: utf-8

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
sys.path.insert(0, '/home/kenlee/energy_market_project/scripts/MySQL_scripts/')
from DAM_prices_by_SP import Feature_Processor
from Sequence_Feature_Processor import Sequence_Feature_Processor

class Keras_NN(object):

    def __init__(self, type='MLP'):
        self.type = type
        self.model = None
        self.feature_df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.y_actual = None
        self.y_pred = None
        self.MAPE = None
        self.TheilU1 = None
        self.TheilU2 = None
        if type == 'MLP':
            self.feature_processor = Feature_Processor()
        else:
            self.feature_processor = Sequence_Feature_Processor()

    def query_db(self, start_date, end_date):
        self.feature_processor.query(start_date, end_date)

    def load_data(self, lzhub, model_type, scaling_method='standard'):
        self.feature_df = self.feature_processor.construct_feature_vector_matrix(lzhub, model_type)
        if self.type == 'MLP':
            self.train_df, self.val_df, self.test_df = self.feature_processor.train_test_validate(scaling=scaling_method)
            self.x_train, self.y_train = self.feature_processor.convert_dfs_to_numpy(self.train_df)
            self.x_test, self.y_test = self.feature_processor.convert_dfs_to_numpy(self.test_df)
            self.x_val, self.y_val = self.feature_processor.convert_dfs_to_numpy(self.val_df)
        if self.type != 'MLP':
            self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self.feature_processor.train_test_validate()


    def create_model(self, hidden_layers):
        if self.type == 'MLP':
            self.model = Sequential()
            self.model.add(Dense(hidden_layers, init = 'glorot_uniform', activation = 'tanh', input_dim = self.x_train.shape[1]))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])
        if self.type == 'SimpleRNN':
            self.model = Sequential()
            self.model.add(SimpleRNN(hidden_layers, input_dim = self.x_train.shape[2], input_length = self.x_train.shape[1]))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='RMSprop')
        if self.type == 'LSTM':
            self.model = Sequential()
            self.model.add(LSTM(hidden_layers, input_dim = self.x_train.shape[2], input_length = self.x_train.shape[1]))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        if self.type == 'StackedLSTM':
            self.model = Sequential()
            self.model.add(LSTM(hidden_layers, return_sequences=True, input_dim = self.x_train.shape[1]))
            self.model.add(LSTM(hidden_layers))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')



    def train_model(self, epochs):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), nb_epoch=epochs)

    def predict(self):
        if self.type == 'MLP':
            self.y_pred = self.model.predict(self.x_test)
            self.y_actual = self.feature_processor.inverse_scale_testing()
            self.y_pred = self.feature_processor.inverse_scale_prediction(self.y_pred)
        if self.type != 'MLP':
            self.y_pred = self.model.predict(self.x_test)
            self.y_actual = self.y_test
            self.y_pred = self.feature_processor.inverse_scale(self.y_pred)
            self.y_actual = self.feature_processor.inverse_scale(self.y_actual)

    def compute_metrics(self):
        self.MAPE = np.sum(np.divide(np.abs(self.y_actual - self.y_pred), self.y_actual)*100)/self.y_actual.shape[0]
        numerator = np.sqrt(np.sum(np.square(self.y_actual - self.y_pred))/self.y_actual.shape[0])
        denominatorA = np.sqrt(np.sum(np.square(self.y_actual))/self.y_actual.shape[0])
        denominatorB = np.sqrt(np.sum(np.square(self.y_pred))/self.y_pred.shape[0])
        self.TheilU1 = numerator/(denominatorA + denominatorB)
        self.TheilU2 = numerator/(denominatorA)

    def plot_results(self):
        plt.plot(self.y_actual, label='actual')
        plt.plot(self.y_pred, label='predicted')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    kMLP = Keras_NN(type='SimpleRNN')
    kMLP.query_db('2012-01-01', '2012-12-31')
    kMLP.load_data('LZ_NORTH', 'C')
    kMLP.create_model(hidden_layers=30)
    kMLP.train_model(epochs=50)
    kMLP.predict()
    kMLP.compute_metrics()
    print(kMLP.MAPE)
    print(kMLP.TheilU1)
    print(kMLP.TheilU2)
    kMLP.plot_results()






