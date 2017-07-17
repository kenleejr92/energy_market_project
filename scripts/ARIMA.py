
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ercot_data_interface import ercot_data_interface
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


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

    def __init__(self, p=2, d=24, q=2, log_difference=True):
        self.linear_regression = None
        self.d = d
        self.q = q
        self.p = p
        self.log_difference = log_difference
        if self.log_difference == True:
            self.d = 1

    def make_lag_matrix(self, v, p):
        X = []
        if p > 1:
            for k in np.arange(1, p):
                v_k= v[(p-k):-k]
                X.append(v_k)
        X.append(v[:-p])
        X = np.array(X)
        X = np.squeeze(X)
        if p >1: 
            X = np.swapaxes(X, 0, 1)
        else:
            X = np.expand_dims(X, 1)
        return X

    def fit(self, x):
        self.linear_regression = LinearRegression()
        if self.log_difference == True:
            self.scaler = MinMaxScaler((1, np.max(x)))
            x = self.scaler.fit_transform(x)
            v = np.log(x[self.d:]) - np.log(x[:-self.d])
        else:
            v = x[self.d:] - x[:-self.d]
        X = self.make_lag_matrix(v, self.p)
        y = np.squeeze(v[self.p:])
        self.linear_regression.fit(X, y)


    def predict(self, x):
        if self.log_difference == True:
            self.scaler = MinMaxScaler((1, np.max(x)))
            x = self.scaler.fit_transform(x)
            v = np.log(x[self.d:]) - np.log(x[:-self.d])
        else:
            v = x[self.d:] - x[:-self.d]
        y = np.squeeze(v[self.p:])
        X = self.make_lag_matrix(v, self.p)
        if self.q == 0:
            y_hat = self.linear_regression.predict(X)
            y_hat = np.expand_dims(y_hat, 1)
            if self.log_difference == True:
                return y_hat, y[self.p:]
            else:
                p_hat = y_hat + x[self.p:-self.d]
                return p_hat, x[self.d + self.p:]
        else:
            y_hat = self.linear_regression.predict(X)
            errors = y_hat - y
            E = self.make_lag_matrix(errors, self.q)/self.q
            error_term = np.sum(E, axis=1)
            y_hat = y_hat[self.q:] + error_term
            y_hat = np.expand_dims(y_hat, 1)
            if self.log_difference == True:
                return y_hat, y[self.p:]
            else:
                p_hat = y_hat + x[self.p+self.q:-self.d]
                return p_hat, x[self.d + self.p + self.q:]

    def plot_predicted_vs_actual(self, x):
        p_hat, z = self.predict(x)
        plt.plot(z, label='actual', color='r')
        plt.plot(p_hat, label='predicted', color='b')
        plt.legend()
        plt.show()


    def print_statistics(self, predicted, actual):
        if len(actual.shape)==1:
            actual = np.expand_dims(actual, 1)
        mae = np.mean(np.abs(predicted - actual))
        trivial = np.mean(np.abs(actual[1:] - actual[:-1]))
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
    crr_nodes = ercot.get_CRR_nodes()
    sources_sinks = ercot.get_sources_sinks()
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[5])
    node0 = ercot.all_nodes[0]
    train, test = ercot.get_train_test(node0, normalize=False, include_seasonal_vectors=False)
    
    arima = ARIMA(p=2, d=1, q=2, log_difference=False)
    arima.fit(train)
    arima.plot_predicted_vs_actual(test)
    predicted, actual = arima.predict(test)
    arima.print_statistics(predicted, actual)
