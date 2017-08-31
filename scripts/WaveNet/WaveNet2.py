import sys
from WaveNet import weight_variable
sys.path.append('..')
from ercot_data_interface import ercot_data_interface
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cPickle as pickle
from ops import * 
from MLP import MLP
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from ARIMA import ARIMA


#input = cut off last sample of time series
#output is length of input - receptive field
#compare output with time series[receptive_field+1:]

def weight_variable(shape, Name, seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed, name=Name)
    return tf.Variable(initial, name=Name)


def bias_variable(shape, Name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=Name)


def offset_variable(shape, Name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=Name)


def scale_variable(shape, Name):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial, name=Name)

class WaveNet2(object):

    def __init__(self, initial_filter_width, 
                        filter_width, 
                        dilation_channels, 
                        dilations,
                        forecast_horizon,
                        random_seed):
        self.initial_filter_width = initial_filter_width
        self.filter_width = filter_width
        self.dilation_channels = dilation_channels
        self.dilations = dilations
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.random_seed = random_seed
        self.forecast_horizon = forecast_horizon

    def calculate_receptive_field(self, filter_width, dilations):
        '''
        Final receptive field for model
        '''
        receptive_field = (filter_width - 1) * sum(dilations) + self.initial_filter_width

        print 'Receptive Field Size:', receptive_field
        return receptive_field

    def create_placeholders(self):
        train_place_holder = tf.placeholder('float', [None, None, self.input_channels], name='input_sequence')
        return train_place_holder


    def create_stack_variables(self):
        var = dict()

        ########################Initial layer##############################
        with tf.variable_scope('causal_layer'):
            layer = dict()
            layer['filter'] = []
            for i in range(self.input_channels):
                layer['filter'].append(weight_variable([self.initial_filter_width, 1, self.dilation_channels], 'filter', self.random_seed))
            var['causal_layer'] = layer


        #######################Dilated Stack###############################
        var['dilation_layer'] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.dilation_channels, self.dilation_channels], 'filter', self.random_seed)
                    var['dilation_layer'].append(current)


        #########################Residual Layer#############################
        var['residual_layer'] = list()
        with tf.variable_scope('residual_layer'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.dilation_channels, self.dilation_channels], 'filter', self.random_seed)
                    var['residual_layer'].append(current)

        ########################Linear Layer#################################
        with tf.variable_scope('linear_layer'):
            layer = dict()
            layer['linear_filter'] = weight_variable([1, self.dilation_channels, self.output_channels], 'linear', self.random_seed)
            var['linear'] = layer

        return var


    def create_causal_layer(self, input):
        self.stack_variables = self.create_stack_variables()
        conditional_filters = []
        for i in range(self.input_channels):
            causal_layer_filter = self.stack_variables['causal_layer']['filter'][i]
            conditional_filters.append(causal_conv(tf.expand_dims(input[:, :, i], axis=2), causal_layer_filter, 1))
        current = tf.add_n(conditional_filters)
        return current


    def create_dilation_stack(self, input, layer_index, dilation):
        self.stack_variables = self.create_stack_variables()

        dilation_filter = self.stack_variables['dilation_layer'][layer_index]['filter']
        current = causal_conv(input, dilation_filter, dilation)
        current = tf.nn.relu(current)
        residual_weights = self.stack_variables['residual_layer'][layer_index]['filter']
        residual_output = tf.nn.conv1d(current, residual_weights, stride=1, padding="SAME", name="residual")
        input_cut = tf.shape(input)[1] - tf.shape(current)[1]
        input = input[:, input_cut:, :]
        return input + residual_output


    def create_network(self, input):
        current = input
        current = self.create_causal_layer(current)
        current = tf.nn.relu(current)
        for i, d in enumerate(self.dilations):
            current = self.create_dilation_stack(current, i, d)
        linear_filter = self.stack_variables['linear']['linear_filter']
        output = tf.nn.conv1d(current, linear_filter, stride=1, padding="SAME")

        return output

    def loss(self, input):
        #cutoff last sample
        output = self.create_network(input[:, :-self.forecast_horizon, :])
        target_output = tf.expand_dims(input[:, self.receptive_field+self.forecast_horizon-1:, 0], axis=2)
        loss = tf.abs(output-target_output)

        return loss, output, target_output


    def train(self, time_series, epochs):
        self.input_channels = time_series.shape[1]
        self.output_channels = 1
        train_stop = int(0.8*time_series.shape[0])
        train = time_series[:train_stop]
        test = time_series[train_stop-self.receptive_field:]
        train = np.pad(time_series, ((self.receptive_field, 0), (0,0)), 'constant', constant_values=((0, 0), (0,0)))
        
        #expand dimensions for batch
        train = np.expand_dims(train, 0)
        test = np.expand_dims(test, 0)

        tf_session = tf.Session()
        x_ = self.create_placeholders()
        self.mae, self.output, self.target_output = self.loss(x_)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.mae)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)

        for e in range(epochs):
            print 'step{}:'.format(e) 
            y_pred, y, _ = tf_session.run([self.output, self.target_output, train_step], feed_dict={x_: train})
            print_statistics(y_pred[0], y[0], self.forecast_horizon)

        y_pred, y, _ = tf_session.run([self.output, self.target_output, train_step], feed_dict={x_: test})
        print_statistics(y_pred[0], y[0], self.forecast_horizon)
        # plot_predicted_vs_actual(y_pred[0], y[0])
        tf_session.close()
        return y_pred[0], y[0]


def plot_predicted_vs_actual(predicted, actual):
    plt.plot(actual, label='True')
    plt.plot(predicted, label='WaveNet')
    plt.xlabel('Hour')
    plt.ylabel('$/MWh')
    plt.legend()
    plt.show()

def print_statistics(predicted, actual, forecast_horizon):
    mae = np.mean(np.abs(predicted - actual))
    trivial = np.mean(np.abs(actual[forecast_horizon:] - actual[:-forecast_horizon]))
    print 'MAE:', mae
    print 'Trivial MAE:', trivial
    mase = mae/trivial
    print 'MASE:', mase
    return mae, mase


def align_time_series(time_series):
    lengths = [p.shape[0] for p in time_series]
    val, idx = min((val, idx) for (idx, val) in enumerate(lengths))
    aligned = []
    for i, p in enumerate(time_series):
        if i==idx: 
            aligned.append(p)
        else:
            cut = p.shape[0] - val
            aligned.append(p[cut:])
    return aligned


if __name__ == '__main__':
    TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']
    test_node = TEST_NODES[1]
    with open('/mnt/hdd1/ERCOT/' + test_node + '_train.pkl', 'r') as f2:
        train = pickle.load(f2)
    with open('/mnt/hdd1/ERCOT/' + test_node + '_test.pkl', 'r') as f2:
        test = pickle.load(f2)
    train_a = np.expand_dims(train[:, 0], 1)
    test_a = np.expand_dims(test[:, 0], 1)
    series = np.vstack((train_a, test_a))
    mimo_series = np.vstack((train, test))
    # scaler = MinMaxScaler((1, np.max(series)))
    # series = scaler.fit_transform(series)
    # series = np.log(series[1:]) - np.log(series[:-1])
    # scaler = MinMaxScaler((1, np.max(mimo_series)))
    # mimo_series = scaler.fit_transform(mimo_series)
    # mimo_series = np.log(mimo_series[1:]) - np.log(mimo_series[:-1])

    wavenet2 = WaveNet2(initial_filter_width=48, 
                        filter_width=2, 
                        dilation_channels=32, 
                        dilations=[1, 2, 4, 8, 16, 32, 64],
                        forecast_horizon=24,
                        random_seed=22943)

    predicted1, actual1 = wavenet2.train(series, 1000)
    mlp = MLP(look_back=175, random_seed=1234, log_difference=False, forecast_horizon=24)
    predicted2, actual2 = mlp.train(mimo_series, epochs=1000)
    


    arima = ARIMA(p=175, d=24, q=174, log_difference=False)
    arima.fit(train_a)
    predicted3, actual3 = arima.predict(test_a)
    print predicted1.shape, predicted2.shape

    aligned = align_time_series([actual1, predicted1, predicted2, predicted3])
   
    plt.plot(aligned[0], label='True')
    plt.plot(aligned[1], label='WaveNet')
    plt.plot(aligned[2], label='MLP')
    # plt.plot(aligned[3], label='ARIMA')
    
    mae_wavenet = np.sqrt(np.mean(np.square(aligned[0]-aligned[1])))
    mae_mlp = np.sqrt(np.mean(np.square(aligned[0]-aligned[2])))
    mae_arima = np.sqrt(np.mean(np.square(aligned[0]-aligned[3])))
    print mae_wavenet, mae_mlp, mae_arima
    plt.legend()
    plt.show()