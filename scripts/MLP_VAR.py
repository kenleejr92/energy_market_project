import numpy as np
from ercot_data_interface import ercot_data_interface
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class MLP_VAR(object):
    def __init__(self, random_seed, log_difference=True, forecast_horizon=1):
        self.forecast_horizon = forecast_horizon
        self.random_seed = random_seed
        self.log_difference = log_difference
        self.MLP = MLPRegressor(random_state = self.random_seed, hidden_layer_sizes=(100,), activation="relu", shuffle=False, batch_size=1024, max_iter=200)
        self.train_fraction = 0.8

    def train(self, time_series, look_back=2):
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
               lags.append(ts[i-k, :])
           X.append(lags)
           y.append(ts[i+self.forecast_horizon-1, :])
        
        X = np.squeeze(np.array(X))
        train_stop = int(self.train_fraction*X.shape[0])
        X = X[:train_stop, :]
        y =np.array(y[:train_stop])
        X = X.reshape((X.shape[0], self.look_back*X.shape[2]))
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
               lags.append(ts[i-k, :])
           X.append(lags)
           y.append(ts[i+self.forecast_horizon-1, :])
        X = np.array(X)
        X = X.reshape((X.shape[0], self.look_back*X.shape[2]))
        y = np.array(y)
        y_pred = self.MLP.predict(X)
        return y_pred, y

    def plot_predicted_vs_actual(self, predicted, actual):
        plt.plot(actual, label='actual', color='r')
        plt.plot(predicted, label='predicted', color='b')
        plt.legend()
        plt.show()

    def print_statistics(self, predicted, actual):
        mae = np.mean(np.abs(predicted - actual))
        trivial = np.mean(np.abs(actual[self.forecast_horizon:] - actual[:-self.forecast_horizon]))
        print 'MAE:', mae
        print 'Trivial MAE:', trivial
        mase = mae/trivial
        print 'MASE:', mase
        hits = (predicted[1:] - predicted[:-1])*(actual[1:] - actual[:-1]) > 0
        HITS = np.mean(np.abs(predicted[1:][hits]))
        print 'HITS:', HITS
        return mae, mase, HITS
    
    

if __name__ == '__main__':
    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    node0 = ercot.all_nodes[5]
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[100])
    train, test = ercot.get_train_test(nn, normalize=False, include_seasonal_vectors=False)
    mlp = MLP_VAR(random_seed=1234, log_difference=True, forecast_horizon=1)
    mlp.train(train, look_back=111)
    predicted, actual = mlp.predict(test)
    mlp.print_statistics(predicted, actual)
    mlp.plot_predicted_vs_actual(predicted, actual)

