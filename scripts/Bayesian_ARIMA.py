import pymc3 as pm
from theano import scan, shared
import numpy as np
from ercot_data_interface import ercot_data_interface
from sklearn.preprocessing import MinMaxScaler


class Bayesian_ARIMA(object):
    def build_model(self, time_series):
        with pm.Model() as arma_model:
            self.scaler = MinMaxScaler((1, np.max(time_series)))
            time_series = self.scaler.fit_transform(time_series)
            time_series = np.log(time_series[1:]) - np.log(time_series[:-1])
            time_series = np.squeeze(time_series)
            y = shared(time_series)
            sigma = pm.HalfCauchy('sigma', 5.)
            theta = pm.Normal('theta', 0., sd=2.)
            phi = pm.Normal('phi', 0., sd=2.)
            mu = pm.Normal('mu', 0., sd=10.)

            err0 = y[0] - (mu + phi * mu)

            def calc_next(last_y, this_y, err, mu, phi, theta):
                nu_t = mu + phi * last_y + theta * err
                return this_y - nu_t

            err, _ = scan(fn=calc_next,
                          sequences=dict(input=y, taps=[-1, 0]),
                          outputs_info=[err0],
                          non_sequences=[mu, phi, theta])

            pm.Potential('like', pm.Normal.dist(0, sd=sigma).logp(err))

        return arma_model


    def run(self, time_series, n_samples=1000):
        model = self.build_model(time_series)
        with model:
            trace = pm.sample(draws=n_samples)

        burn = n_samples // 10
        pm.plots.traceplot(trace[burn:])
        pm.plots.forestplot(trace[burn:])


if __name__ == '__main__':
    bayesian_arima = Bayesian_ARIMA()
    ercot = ercot_data_interface()
    train, test = ercot.get_train_test('CAL_CALGT1', normalize=False, include_seasonal_vectors=False)
    bayesian_arima.run(train)