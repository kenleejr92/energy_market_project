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
                        use_batch_norm,
                        dilations,
                        random_seed):
        self.initial_filter_width = initial_filter_width
        self.filter_width = filter_width
        self.dilation_channels = dilation_channels
        self.batch_norm = use_batch_norm
        self.dilations = dilations
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.random_seed = random_seed

    def calculate_receptive_field(self, filter_width, dilations):
        '''
        Final receptive field for model
        '''
        receptive_field = (filter_width - 1) * sum(dilations) + self.initial_filter_width

        print 'Receptive Field Size:', receptive_field
        return receptive_field

    def create_placeholders(self):
        train_place_holder = tf.placeholder('float', [None, self.sequence_length, self.input_channels], name='input_sequence')
        return train_place_holder


    def create_stack_variables(self):
        initial_channels = self.output_channels
        var = dict()

        # ########################Batch Normalization#######################
        with tf.variable_scope('batch_normalization'):
            layer = dict()
            layer['filter_scale'] = scale_variable([self.sequence_length-1, self.input_channels], 'BN_scaler')
            layer['filter_offset'] = offset_variable([self.sequence_length-1, self.input_channels], 'BN_offset')
            var['batch_norm'] = layer

        ########################Initial layer##############################
        with tf.variable_scope('causal_layer'):
            layer = dict()
            layer['filter'] = weight_variable([self.initial_filter_width, self.input_channels, self.dilation_channels], 'filter', self.random_seed)
            var['causal_layer'] = layer

        
        #######################Dilated Stack###############################
        var['dilation_layer'] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.dilation_channels, self.dilation_channels], 'filter', self.random_seed)
                    var['dilation_layer'].append(current)


        var['residual_layer'] = list()
        with tf.variable_scope('residual_layer'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.dilation_channels, self.dilation_channels], 'filter', self.random_seed)
                    var['residual_layer'].append(current)

        var['skip_layer'] = list()
        with tf.variable_scope('skip_layer'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.dilation_channels, self.dilation_channels], 'filter', self.random_seed)
                    var['skip_layer'].append(current)


        with tf.variable_scope('linear_layer'):
            layer = dict()
            layer['linear_filter'] = weight_variable([1, self.dilation_channels, self.output_channels], 'linear', self.random_seed)
            var['linear'] = layer

        return var


    def create_causal_layer(self, input):
        self.stack_variables = self.create_stack_variables()
        causal_layer_filter = self.stack_variables['causal_layer']['filter']
        current = causal_conv(input, causal_layer_filter, 1)
        return current


    def create_dilation_stack(self, input, layer_index, dilation, output_width):
        self.stack_variables = self.create_stack_variables()

        if self.batch_norm == True:
            batch_norm_scaler = self.stack_variables['batch_norm']['filter_scale']
            batch_norm_offset = self.stack_variables['batch_norm']['filter_offset']
            batch_mean, batch_var = tf.nn.moments(input, [1])
            input = tf.nn.batch_normalization(input, batch_mean, batch_var, batch_norm_offset, batch_norm_scaler, 0.0001)

        dilation_filter = self.stack_variables['dilation_layer'][layer_index]['filter']
        current = causal_conv(input, dilation_filter, dilation)
        current = tf.nn.relu(current)
        residual_weights = self.stack_variables['residual_layer'][layer_index]['filter']
        residual_output = tf.nn.conv1d(current, residual_weights, stride=1, padding="SAME", name="residual")
        skip_weights = self.stack_variables['skip_layer'][layer_index]['filter']
        skip_contrib = tf.nn.conv1d(current, skip_weights, stride=1, padding="SAME", name="skip")
        skip_cut = tf.shape(current)[1] - output_width
        skip_contrib = skip_contrib[:, skip_cut:, :]
        input_cut = tf.shape(input)[1] - tf.shape(current)[1]
        input = input[:, input_cut:, :]
        return skip_contrib, input + residual_output


    def create_network2(self, input):
        current = input
        output_width = tf.shape(current[:, self.receptive_field-1:, :])[1]
        skip_contribs = []
        current = self.create_causal_layer(current)
        for i, d in enumerate(self.dilations):
            skip_contrib, current = self.create_dilation_stack(current, i, d, output_width)
            skip_contribs.append(skip_contrib)
        skip_contribs = tf.stack(skip_contribs, axis=0)
        output = tf.reduce_sum(skip_contribs, axis=[0])
        linear_filter = self.stack_variables['linear']['linear_filter']
        output = tf.nn.conv1d(output, linear_filter, stride=1, padding="SAME")

        return output

    def create_network(self, input):
        self.stack_variables = self.create_stack_variables()

        if self.batch_norm == True:
            batch_norm_scaler = self.stack_variables['batch_norm']['filter_scale']
            batch_norm_offset = self.stack_variables['batch_norm']['filter_offset']
            batch_mean, batch_var = tf.nn.moments(input, [1])
            input = tf.nn.batch_normalization(input, batch_mean, batch_var, batch_norm_offset, batch_norm_scaler, 0.0001)

        causal_layer_filter = self.stack_variables['causal_layer']['filter']
        current = causal_conv(input, causal_layer_filter, 1)
       

        for layer_index, dilation in enumerate(self.dilations):
            dilation_filter = self.stack_variables['dilation_layer'][layer_index]['filter']
            current = causal_conv(current, dilation_filter, dilation)
        current = tf.nn.relu(current)

        linear_filter = self.stack_variables['linear']['linear_filter']

        
        output = tf.nn.conv1d(current, linear_filter, stride=1, padding="SAME")

        return output

    def loss(self, input):
        #cutoff last sample
        output = self.create_network(input[:, :-1, :])
        target_output = input[:, self.receptive_field:, :]
        loss = tf.reduce_mean(tf.abs(output-target_output))

        return loss, output, target_output


    def train(self, time_series, epochs):
        self.input_channels = time_series.shape[1]
        self.output_channels = time_series.shape[1]
        time_series = np.pad(time_series, ((self.receptive_field, 0), (0,0)), 'constant', constant_values=((0, 0), (0,0)))
        train_stop = int(0.8*time_series.shape[0])
        train = time_series[:train_stop]
        print train.shape
        test = time_series[train_stop:]
        self.sequence_length = train.shape[0]
        train = np.expand_dims(train, 0)

        history = np.expand_dims(time_series, 0)

        tf_session = tf.Session()
        x_ = self.create_placeholders()
        self.mae, self.output, self.target_output = self.loss(x_)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.mae)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)

        for e in range(epochs):
            print 'step{}:'.format(e) 
            y_pred, y, _ = tf_session.run([self.output, self.target_output, train_step], feed_dict={x_: train})
            print_statistics(y_pred[0], y[0])

        test_pred = []
        for i in range(history.shape[1]):
            if history[:, i+1:i+1+self.sequence_length, :].shape[1] != self.sequence_length: 
                break
            y_pred, y = tf_session.run([self.output, self.target_output], feed_dict={x_: history[:, i+1:i+1+self.sequence_length, :]})
            test_pred.append(y_pred[0][-1, :])

        test_pred = np.array(test_pred).reshape(-1, 1)
        mae, mase, hits = print_statistics(test_pred, test)
        plot_predicted_vs_actual(test_pred, test)
        tf_session.close()
        return mae, mase, hits



def plot_predicted_vs_actual(predicted, actual):
    plt.plot(actual[211:], label='True')
    print 'wavenet length', predicted.shape[0]
    plt.plot(predicted[211:], label='WaveNet')
    plt.xlabel('Hour')
    plt.ylabel('$/MWh')
    plt.legend()
    plt.show()

def print_statistics(predicted, actual):
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
    TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']
    test_node = TEST_NODES[0]
    with open('/mnt/hdd1/ERCOT/' + test_node + '_train.pkl', 'r') as f2:
        train = pickle.load(f2)
    with open('/mnt/hdd1/ERCOT/' + test_node + '_test.pkl', 'r') as f2:
        test = pickle.load(f2)
    train_a = np.expand_dims(train[:, 0], 1)
    test_a = np.expand_dims(test[:, 0], 1)
    series = np.vstack((train_a, test_a))
    mimo_series = np.vstack((train, test))
    
    scaler = MinMaxScaler((1, np.max(series)))
    series = scaler.fit_transform(series)
    series = np.log(series[1:]) - np.log(series[:-1])

    wavenet2 = WaveNet2(initial_filter_width=48, 
                        filter_width=2, 
                        dilation_channels=32, 
                        use_batch_norm=False,
                        dilations=[1, 2, 4, 8, 16, 32, 64],
                        random_seed=22943)

    mlp = MLP(random_seed=1234, log_difference=True, forecast_horizon=1)
    mlp.train(train_a, look_back=175)
    predicted, actual = mlp.predict(test_a)
    plt.plot(predicted, label='MLP')
    print 'MLP length', predicted.shape[0]

    arima = ARIMA(p=2, d=24, q=2, log_difference=True)
    arima.fit(train_a)
    predicted, actual = arima.predict(test_a)
    plt.plot(predicted[148:], label='ARIMA')
    print 'ARIMA length', predicted.shape[0]

    wavenet2.train(series, 5000)

