import numpy as np
from ercot_data_interface import ercot_data_interface
from sklearn.neural_network import MLPRegressor


class MLP(object):
    def __init__(self):
        self.MLP = MLPRegressor()


    def train(self, time_series, look_back=24):
        self.look_back = look_back
        X = []
        y = []
        for i in np.arange(self.look_back + 1, x.shape[0]):
           lags = []
           for k in np.arange(1, self.look_back + 1):
               lags.append(x[i-k])
           X.append(lags)
           y.append(x[i])
        X = np.squeeze(np.array(X))
        y = np.array(y)
        self.MLP.fit(X, y)


    def predict(self, time_series)
        X = []
        y = []
        for i in np.arange(self.look_back + 1, t.shape[0]):
           lags = []
           for k in np.arange(1, self.look_back+1):
               lags.append(t[i-k])
           X.append(lags)
           y.append(t[i])
        X = np.squeeze(np.array(X))
        y = np.array(y)
        y_pred = MLP.predict(X)
        print np.mean(np.abs(y_pred-y))
    

    if __name__ == '__main__':
        pass