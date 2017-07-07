import numpy as np
from ercot_data_interface import ercot_data_interface
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self, log_difference=True, forecast_horizon=1):
        self.forecast_horizon = forecast_horizon
        self.log_difference = log_difference
        self.MLP = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", batch_size=1024, max_iter=200)
        self.train_fraction = 0.8

    def train(self, time_series, look_back=257):
        if self.log_difference == True:
            self.scaler = MinMaxScaler((1, np.max(time_series)))
            time_series = self.scaler.fit_transform(time_series)
            ts = np.log(time_series[1:]) - np.log(time_series[:-1])
            self.train_past = np.log(time_series[:-1])
        else:
            ts = time_series
        
        self.look_back = look_back
        X = []
        y = []
        for i in np.arange(self.look_back + 1, ts.shape[0]):
           lags = []
           if i + self.forecast_horizon - 1 >= ts.shape[0]: break
           for k in np.arange(1, self.look_back + 1):
               lags.append(ts[i-k])
           X.append(lags)
           y.append(ts[i+self.forecast_horizon-1])
        X = np.squeeze(np.array(X))
        train_stop = int(self.train_fraction*X.shape[0])
        X = X[train_stop:]
        y = np.reshape(np.array(y[train_stop:]), (-1,))
        self.MLP.fit(X, y)


    def predict(self, time_series):
        if self.log_difference == True:
            scaler = MinMaxScaler((1, np.max(time_series)))
            time_series = scaler.fit_transform(time_series)
            ts = np.log(time_series[1:]) - np.log(time_series[:-1])
            self.test_past = np.log(time_series[:-1])
        else:
            ts = time_series
        X = []
        y = []
        for i in np.arange(self.look_back + 1, ts.shape[0]):
           lags = []
           if i + self.forecast_horizon - 1 >= ts.shape[0]: break
           for k in np.arange(1, self.look_back+1):
               lags.append(ts[i-k])
           X.append(lags)
           y.append(ts[i+self.forecast_horizon-1])
        X = np.squeeze(np.array(X))
        y = np.array(y)
        y_pred = self.MLP.predict(X)
        y_pred = np.reshape(y_pred, (-1, 1))
        y = np.reshape(y, (-1, 1))
        
        if self.log_difference == True:
            y_pred = np.exp(y_pred + self.test_past[self.look_back + self.forecast_horizon:])
            y = np.exp(y + self.test_past[self.look_back + self.forecast_horizon:])
            y_pred = self.scaler.inverse_transform(y_pred)
            y = self.scaler.inverse_transform(y)
        print y_pred.shape
        mae = np.mean(np.abs(y_pred-y))
        trivial = np.mean(np.abs(y[self.forecast_horizon:] - y[:-self.forecast_horizon]))
        print 'MAE:', mae
        print 'MASE:', mae/trivial
        plt.plot(y_pred, label='predicted', color='b')
        plt.plot(y, label='actual', color='r')
        plt.legend()
        plt.show()

        
    

if __name__ == '__main__':
    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    node0 = ercot.all_nodes[0]
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[100])
    train, test = ercot.get_train_test(node0, normalize=False, include_seasonal_vectors=False)
    mlp = MLP(forecast_horizon=2000)
    mlp.train(train)
    mlp.predict(test)