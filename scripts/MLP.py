
import numpy as np
from ercot_data_interface import ercot_data_interface
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


def weight_variable(shape, Name, seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed, name=Name)
    return tf.Variable(initial, name=Name)

def bias_variable(shape, Name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=Name)

class MLP(object):
    def __init__(self, look_back, random_seed, log_difference=True, forecast_horizon=1):
        self.forecast_horizon = forecast_horizon
        self.log_difference = log_difference
        self.random_seed = random_seed
        self.MLP = MLPRegressor(random_state = self.random_seed, hidden_layer_sizes=(100,), activation="relu", shuffle=False, batch_size=1024, max_iter=5000)
        self.train_fraction = 1
        self.hidden_layers1 = 100
        self.hidden_layers2 = 100
        self.look_back = look_back


    def create_place_holders(self):
        X_ = tf.placeholder('float', [None, self.num_features], name='input_sequence')
        y_ = tf.placeholder('float', [None, 1], name='target_sequence')
        return X_, y_

    def create_variables(self):
        self.w1 = weight_variable([self.num_features, self.hidden_layers1], 'hidden1', seed=self.random_seed)
        self.w2 = weight_variable([self.hidden_layers1, self.hidden_layers2], 'hidden2', seed=self.random_seed)
        self.out = weight_variable([self.hidden_layers1, 1], 'out', seed=self.random_seed)
        self.b1 = bias_variable([self.hidden_layers1], 'bias1')
        self.b2 = bias_variable([self.hidden_layers2], 'bias2')
        self.b3 = bias_variable([1], 'bias3')

    def make_lag_matrix(self, ts):
        X = []
        y = []
        for i in np.arange(self.look_back + 1, ts.shape[0]):
           lags = []
           if i + self.forecast_horizon - 1 >= ts.shape[0]: 
               break
           for k in np.arange(1, self.look_back + 1):
               lags.append(ts[i-k])
           lags = np.array(lags)
           lags = np.reshape(lags, (-1, 1))
           X.append(lags)
           y.append(ts[i+self.forecast_horizon-1, 0])
        X = np.squeeze(np.array(X))
        y = np.reshape(np.array(y), (-1,))
        return X, y


    def create_network(self, input):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(input, self.w1), self.b1))
        out_layer = tf.matmul(layer_1, self.out) + self.b3
        return out_layer

    def loss(self, input, target):
        output = self.create_network(input)
        mse = tf.reduce_mean(tf.square(output-target))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss = (mse + 0.001 * l2_loss)
        return total_loss, output, target


    def train(self, time_series, epochs):
        X, y = self.make_lag_matrix(time_series)
        # self.num_features = X.shape[1]
        # self.create_variables()
        # X_, y_ = self.create_place_holders()
        # self.mse, self.output, self.target = self.loss(X_, y_)
        # self.batch_size = 2056
        # tf_session = tf.Session()
        # self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.mse)
        # init_op = tf.global_variables_initializer()
        # tf_session.run(init_op)
        train_stop = int(0.8*time_series.shape[0])
        train_x = X[:train_stop]
        train_y = y[:train_stop]
        test_x = X[train_stop:]
        test_y = y[train_stop:]
        # for e in range(epochs):
        #     print 'step{}:'.format(e) 
        #     for i in range(int(train_x.shape[0]/self.batch_size)):
        #         y_pred, y, _ = tf_session.run([self.output, self.target, self.train_step], feed_dict={X_: train_x[i:i+self.batch_size], y_: train_y[i:i+self.batch_size]})
        # predicted, actual = tf_session.run([self.output, self.target], feed_dict={X_: test_x, y_: test_y})
        self.MLP.fit(train_x, train_y)
        predicted = self.MLP.predict(test_x)
        predicted = np.reshape(predicted, (-1, 1))
        actual = np.reshape(test_y, (-1, 1))
        return predicted, actual
        # print 'Got data'
        # self.MLP.fit(X, y)
        # print 'Fit done'


    def predict(self, time_series):
        if self.log_difference == True:
            scaler = MinMaxScaler((1, np.max(time_series)))
            time_series = scaler.fit_transform(time_series)
            ts = np.log(time_series[1:]) - np.log(time_series[:-1])
            self.test_past = np.log(time_series[:-1])
        else:
            ts = time_series
        X, y = self.make_lag_matrix(time_series)
        print 'Got data'
        y_pred = self.MLP.predict(X)
        print 'Prediction Done'
        y_pred = np.reshape(y_pred, (-1, 1))
        y = np.reshape(y, (-1, 1))
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
    train, test = ercot.get_train_test(node0, normalize=False, include_seasonal_vectors=False)
    mlp = MLP(random_seed=1234, look_back=175, log_difference=False, forecast_horizon=1)
    predicted, actual = mlp.train(train, epochs=5)
    mlp.print_statistics(predicted, actual)
    mlp.plot_predicted_vs_actual(predicted, actual)

