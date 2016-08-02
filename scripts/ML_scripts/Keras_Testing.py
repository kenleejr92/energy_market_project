
# coding: utf-8

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
sys.path.insert(0, '/home/kenlee/energy_market_project/scripts/MySQL_scripts/')
from DAM_prices_by_SP import Feature_Processor

class Keras_NN(object):

    def __init__(self):
        self.feature_processor = Feature_Processor()
        self.model = None
        self.feature_df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.x_train = None
        self.y_train = None
        self.x_test= None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.y_actual = None
        self.y_pred = None
        self.MAPE = None
        self.TheilU1 = None
        self.TheilU2 = None

    def load_data(self, start_date, end_date, lzhub, model_type):
        self.feature_processor.query(start_date, end_date)
        self.feature_df = self.feature_processor.construct_feature_vector_matrix(lzhub, model_type)
        self.train_df, self.val_df, self.test_df = self.feature_processor.train_test_validate(scaling='standard')
        self.x_train, self.y_train = self.feature_processor.convert_dfs_to_numpy(self.train_df)
        self.x_test, self.y_test = self.feature_processor.convert_dfs_to_numpy(self.test_df)
        self.x_val, self.y_val = self.feature_processor.convert_dfs_to_numpy(self.val_df)


    def create_model(self, hidden_layers, type = 'MLP'):
        if type == 'MLP':
            self.model = Sequential()
            self.model.add(Dense(hidden_layers, init = 'glorot_uniform', activation = 'tanh', input_dim = self.x_train.shape[1]))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])
            self.model.summary()
        if type == 'LSTM':
            self.model = Sequential()
            self.model.add(LSTM(hidden_layers, return_sequences=True, batch_input_shape=(self.x_train.shape[0], 1, self.x_train.shape[1])))
            self.model.add(LSTM(hidden_layers))
            self.model.add(Dense(1, init = 'zero', activation = 'linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
            self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))
            self.x_val = np.reshape(self.x_val, (self.x_val.shape[0], 1, self.x_val.shape[1]))


    def train_model(self, epochs):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), nb_epoch=epochs)

    def predict(self):
        self.y_actual = self.feature_processor.inverse_scale_testing()
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = self.feature_processor.inverse_scale_prediction(self.y_pred)

    def compute_metrics(self):
        self.MAPE = np.sum(np.divide(np.abs(self.y_actual - self.y_pred), self.y_actual)*100)/self.y_actual.shape[0]
        numerator = np.sqrt(np.sum(np.square(self.y_actual - self.y_pred))/self.y_actual.shape[0])
        denominator = np.sqrt(np.sum(np.square(self.y_actual))/self.y_actual.shape[0]) + np.sqrt(np.sum(np.square(self.y_pred))/self.y_pred.shape[0])
        self.TheilU1 = numerator/denominator

    def plot_results(self):
        plt.plot(self.y_actual, label='actual')
        plt.plot(self.y_pred, label='predicted')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    kMLP = Keras_NN()
    kMLP.load_data('2012-01-01', '2012-12-31', 'LZ_HOUSTON', 'LSTM')
    kMLP.create_model(hidden_layers=20, type='LSTM')
    kMLP.train_model(epochs=50)
    kMLP.predict()
    kMLP.compute_metrics()
    print(kMLP.MAPE)
    print(kMLP.TheilU1)
    kMLP.plot_results()






