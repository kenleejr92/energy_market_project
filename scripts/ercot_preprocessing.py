
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ercot_data_interface import ercot_data_interface
from sklearn.linear_model import LinearRegression


def difference(x, lag):
    v = x[lag:] - x[:-lag]
    return v

def undifference(v, x, lag):
    x = v + x[:-lag]
    return x


def autocorrelation(x, lag):
    pt = x[lag:]
    pt_k = x[:-lag]
    mpt = np.mean(pt)
    mpt_k = np.mean(pt_k)
    sdt = np.std(pt)
    sdt_k = np.std(pt_k)
    return np.mean(np.multiply((pt-mpt), (pt_k-mpt_k)))/(sdt*sdt_k)


def plot_autocorrelations(x, max_lag=24):
    autocorrelations = []
    for k in np.arange(1, max_lag):
        autocorrelations.append(autocorrelation(x, k))
    lags = np.arange(1, max_lag)
    plt.bar(lags, autocorrelations)
    plt.xlabel('lag')
    plt.ylabel('correlation coeff')
    plt.show()


class ARIMA(object):

    def __init__(self, p=2, d=0, q=2, seasonal=24):
        self.linear_regression = None
        self.d = d
        self.q = q
        self.p = p
        self.seasonal = seasonal

    def make_lag_matrix(self, v, p):
        X = []
        for k in np.arange(1, p):
            v_k= v[(p-k):-k]
            X.append(v_k)
        X.append(v[:-p])
        X = np.squeeze(np.array(X))
        X = np.swapaxes(X, 0, 1)
        return X

    def fit(self, x):
        self.linear_regression = LinearRegression()
        v = x[self.seasonal:] - x[:-self.seasonal]
        X = self.make_lag_matrix(v, self.p)
        y = np.squeeze(v[self.p:])
        self.linear_regression.fit(X, y)


    def predict(self, x):
        v = x[self.seasonal:] - x[:-self.seasonal]
        y = np.squeeze(v[self.p:])
        X = self.make_lag_matrix(v, self.p)
        if self.q == 0:
            y_hat = self.linear_regression.predict(X)
            y_hat = np.expand_dims(y_hat, 1)
            p_hat = y_hat + x[self.p:-24]
            return p_hat, x[self.seasonal + self.p:]
        else:
            y_hat = self.linear_regression.predict(X)
            errors = y_hat - y
            E = self.make_lag_matrix(errors, self.q)/self.q
            error_term = np.sum(E, axis=1)
            y_hat = y_hat[self.q:] + error_term
            y_hat = np.expand_dims(y_hat, 1)
            p_hat = y_hat + x[self.p+self.q:-24]
            return p_hat, x[self.seasonal + self.p + self.q:]

    def plot_predicted_vs_actual(self, x):
        p_hat, z self.predict(x)
        plt.plot(z, label='actual')
        plt.plot(p_hat, label='predicted')
        plt.legend()
        plt.show()


    def mape(self, x):
        p_hat, z = self.predict(x)
        return np.mean(np.abs(p_hat-z))


if __name__ == '__main__':
    ercot = ercot_data_interface()
    crr_nodes = ercot.get_CRR_nodes()
    # sources_sinks = ercot.get_sources_sinks()
    # nn = ercot.get_nearest_CRR_neighbors(sources_sinks[20])
    x = ercot.query_prices(crr_nodes[1], '2011-01-01', '2014-5-23').as_matrix()
    y = ercot.query_prices(crr_nodes[1], '2014-5-23', '2016-5-23').as_matrix()
    arima = ARIMA(p=5, d=0, q=5, seasonal=24)
    arima.fit(x)
    