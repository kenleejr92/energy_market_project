import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from ercot_data_interface import ercot_data_interface
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from ops import * 
import os
import shutil

def weight_variable(shape, Name):
    initial = tf.truncated_normal(shape, stddev=0.1)
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


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class WaveNet(object):
    '''
    WaveNet for time series
    self.sequence_length: length of sequence input into network
    self.initial_filter_width: filter width for first causal convolution
    self.filter_width: filter width for dilated convolutions
    self.residual_channels: how many filters for parameterized for residual skip connections
    self.dilation_channels: how many filters for each dilation layer
    self.output_channels: how many output channels (1 for single output, n for MIMO)
    self.skip_channels: how many filters for parameterized skip connections
    self.use_biases: whether to use biasing in the network
    self.use_batch_norm: batch normalization before each convolutional layer
    self.dilations : list of dilation factors
    self.num_condition_series: how many series to condition on
    self.histograms: record histograms
    '''
    def __init__(self, forecast_horizon, MIMO=False):
        self.forecast_horizon = forecast_horizon
        self.log_difference = None
        self.epochs = 10
        self.batch_size = 128
        self.initial_filter_width = 2
        self.filter_width = 2
        self.residual_channels = 32
        self.dilation_channels = 32
        self.MIMO = MIMO
        self.skip_channels = 256
        self.use_biases = True
        self.use_batch_norm = False
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.sequence_length = self.receptive_field + 1
        self.histograms = True
        self.random_seed = 22943
        self.create_dirs()

    def create_dirs(self):
        self.log_dir = './' + str(self.random_seed)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.train_log = self.log_dir + '/train'
        self.val_log = self.log_dir + '/val'
        self.save_path = self.log_dir + '/WaveNet'

    def delete_dirs(self):
        shutil.rmtree(self.log_dir)


    def calculate_receptive_field(self, filter_width, dilations):
        '''
        Final receptive field for model
        '''
        receptive_field = (filter_width - 1) * sum(dilations) + self.initial_filter_width
        print 'Receptive Field Size:', receptive_field
        return receptive_field

    def set_parameters(self, time_series):
        if time_series.shape[1] == 1:
            self.output_channels = 1
            self.num_condition_series = None
            self.batch_index = 0
        else:
            if self.MIMO == True:
                self.output_channels = time_series.shape[1]
                self.num_condition_series = None
                self.batch_index = 0
            else:
                self.output_channels = 1
                self.num_condition_series = time_series.shape[1]
                self.batch_index = 1

    def inverse_transform(self, predicted, actual):
        if self.num_condition_series is None:
            predicted = np.exp(predicted + self.past[self.receptive_field+self.forecast_horizon:])
            actual = np.exp(actual + self.past[self.receptive_field+self.forecast_horizon:])
            predicted = self.scaler.inverse_transform(predicted)
            actual = self.scaler.inverse_transform(actual)
            print actual.shape
        elif self.MIMO == False:
            past = self.past[self.receptive_field+self.forecast_horizon:, 0].reshape(-1, 1)
            predicted = np.exp(predicted + past)
            actual = np.exp(actual + past)
            predicted = self.scalers[0].inverse_transform(predicted)
            actual = self.scalers[0].inverse_transform(actual)
        elif self.MIMO == True:
            predicted = np.exp(predicted + self.past[self.receptive_field+self.forecast_horizon:])
            actual = np.exp(actual + self.past[self.receptive_field+self.forecast_horizon:])
            for j in range(num_condition_series):
                predicted = self.scalerd[j].inverse_transform(predicted[:, j])
                actual = self.scalers[j].inverse_transform(actual[:, j])
        return predicted, actual

    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        if self.num_condition_series is None:
            return tf.placeholder('float', [None, self.sequence_length, self.output_channels], name='input_sequence')
        else:
            condition_placeholders = tf.placeholder('float', [self.num_condition_series, None, self.sequence_length, 1], name='condition_sequences')
            return condition_placeholders


    def create_condition_vars(self):
        initial_channels = 1
        var = dict()
        layer = dict()
        var['condition_layer'] = list()
        with tf.variable_scope('causal_layer_par'):
            for j in range(self.num_condition_series):
                with tf.variable_scope('causal_layer'):
                    layer['filter{}'.format(j)] = weight_variable([self.initial_filter_width, initial_channels, self.residual_channels], 'filter{}'.format(j))
                    var['causal_layer'] = layer

                
                with tf.variable_scope('first_dilation'):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.residual_channels, self.dilation_channels], 'filter')
                    current['dense'] = weight_variable([1, self.dilation_channels, self.residual_channels], 'dense')
                    current['skip'] = weight_variable([1, self.dilation_channels, self.skip_channels], 'skip')


                    if self.use_biases:
                        current['filter_bias'] = bias_variable([self.dilation_channels], 'filter_bias')
                        current['dense_bias'] = bias_variable([self.residual_channels], 'dense_bias')
                        current['skip_bias'] = bias_variable([self.skip_channels], 'slip_bias')

                    if self.use_batch_norm:
                        current_receptive_field = self.initial_filter_width
                        current['filter_scale'] = scale_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_scaler')
                        current['filter_offset'] = offset_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_offset')

                    var['condition_layer'].append(current)

        var['dilated_stack'] = list()
        with tf.variable_scope('dilation_stack_par'):        
            for i, dilation in enumerate(self.dilations):
                if i == 0: var['dilated_stack'].append(None)
                with tf.variable_scope('layer{}'.format(i+1)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.residual_channels, self.dilation_channels], 'filter')
                    current['dense'] = weight_variable([1, self.dilation_channels, self.residual_channels], 'dense')
                    current['skip'] = weight_variable([1, self.dilation_channels, self.skip_channels], 'skip')


                    if self.use_biases:
                        current['filter_bias'] = bias_variable([self.dilation_channels], 'filter_bias')
                        current['dense_bias'] = bias_variable([self.residual_channels], 'dense_bias')
                        current['skip_bias'] = bias_variable([self.skip_channels], 'slip_bias')

                    if self.use_batch_norm:
                        current_receptive_field = (self.filter_width-1)*np.sum(self.dilations[:i+1]) + self.initial_filter_width
                        current['filter_scale'] = scale_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_scaler')
                        current['filter_offset'] = offset_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_offset')

                    var['dilated_stack'].append(current)

        with tf.variable_scope('postprocessing_par'):
            current = dict()
            current['postprocess'] = weight_variable([1, self.skip_channels, self.output_channels], 'postprocess')
            if self.use_biases:
                current['postprocess_bias'] = bias_variable([self.output_channels], 'postprocess_bias')
            var['postprocessing'] = current
        return var


    def create_variables(self):
        '''This function creates all variables used by the network in the autoregressive case.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''
        # scalar (univariate) prediction
        initial_channels = self.output_channels
        var = dict()

        with tf.variable_scope('causal_layer'):
            layer = dict()
            layer['filter'] = weight_variable([self.initial_filter_width, initial_channels, self.residual_channels], 'filter')
            var['causal_layer'] = layer

                
        var['dilated_stack'] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.residual_channels, self.dilation_channels], 'filter')
                    current['dense'] = weight_variable([1, self.dilation_channels, self.residual_channels], 'dense')
                    current['skip'] = weight_variable([1, self.dilation_channels, self.skip_channels], 'skip')


                    if self.use_biases:
                        current['filter_bias'] = bias_variable([self.dilation_channels], 'filter_bias')
                        current['dense_bias'] = bias_variable([self.residual_channels], 'dense_bias')
                        current['skip_bias'] = bias_variable([self.skip_channels], 'slip_bias')

                    if self.use_batch_norm:
                        current_receptive_field = (self.filter_width-1)*np.sum(self.dilations[:i]) + self.initial_filter_width
                        current['filter_scale'] = scale_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_scaler')
                        current['filter_offset'] = offset_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_offset')
                        

                    var['dilated_stack'].append(current)

        with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess'] = weight_variable([1, self.skip_channels, self.output_channels], 'postprocess')
                if self.use_biases:
                    current['postprocess_bias'] = bias_variable([self.output_channels], 'postprocess_bias')
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        if self.num_condition_series is None:
            with tf.name_scope('causal_layer'):
                weights_filter = self.variables['causal_layer']['filter']
                output = causal_conv(input_batch, weights_filter, 1)
                return output
        else:
            output = []
            for j in range(self.num_condition_series):
                with tf.name_scope('causal_layer{}'.format(j)):
                    weights_filter = self.variables['causal_layer']['filter{}'.format(j)]
                    output.append(causal_conv(input_batch[j, :, :, :], weights_filter, 1))
            return output

    def _create_dilation_layer(self, input_batch, layer_index, dilation, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
                 |-----------------------------------------------------------------------|
                 |                                                                       |
        condition|-> [BNx]->[filterx] -|           |-> 1x1 conv -> skip output           |
                 |                     |-> (+)-ReLu|----------------------------------->(+)-> 1x1 conv -> residual output
        input -  |-> [BN0]->[filter0] -|                                                 |
                 |                                                                       |
                 |-----------------------------------------------------------------------|

        sum(skip_outputs) -> 1x1 conv --> output

        Where `[filter]` are causal convolutions with a
        ReLu at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        if self.num_condition_series is None or layer_index > 0:
            variables = self.variables['dilated_stack'][layer_index]
            weights_filter = variables['filter']

            if self.use_batch_norm:
                filter_scaler = variables['filter_scale']
                filter_offset = variables['filter_offset']
                batch_mean, batch_var = tf.nn.moments(input_batch, [0])
                batch_normalized = tf.nn.batch_normalization(input_batch, batch_mean, batch_var, filter_offset, filter_scaler, 0.001)
                conv_filter = causal_conv(batch_normalized, weights_filter, dilation)
            else:
                conv_filter = causal_conv(input_batch, weights_filter, dilation)
            

            if self.use_biases:
                filter_bias = variables['filter_bias']
                conv_filter = tf.add(conv_filter, filter_bias)
        else:
            conv_filters = []
            for j in range(self.num_condition_series):
                variables = self.variables['condition_layer'][j]
                weights_filter = variables['filter']

                if self.use_batch_norm:
                    filter_scaler = variables['filter_scale']
                    filter_offset = variables['filter_offset']
                    batch_mean, batch_var = tf.nn.moments(input_batch[j], [0])
                    batch_normalized = tf.nn.batch_normalization(input_batch[j], batch_mean, batch_var, filter_offset, filter_scaler, 0.001)
                    conv_filter = causal_conv(batch_normalized, weights_filter, dilation)
                else:
                    conv_filter = causal_conv(input_batch[j], weights_filter, dilation)

                if self.use_biases:
                    filter_bias = variables['filter_bias']
                    conv_filter = tf.add(conv_filter, filter_bias)
                conv_filters.append(conv_filter)
            conv_filter = sum(conv_filters)

        # The 1x1 conv to produce the residual output
        out = tf.nn.relu(conv_filter)

        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.summary.histogram(layer + '_filter', weights_filter)
            tf.summary.histogram(layer + '_dense', weights_dense)
            tf.summary.histogram(layer + '_skip', weights_skip)
            if self.use_batch_norm:
                tf.summary.histogram(layer + '_BNscaler', filter_scaler)
                tf.summary.histogram(layer + '_BNoffset', filter_offset)
            if self.use_biases:
                tf.summary.histogram(layer + '_biases_filter', filter_bias)
                tf.summary.histogram(layer + '_biases_dense', dense_bias)
                tf.summary.histogram(layer + '_biases_skip', skip_bias)

        if self.num_condition_series is None or layer_index > 0:
            input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
            input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
            return skip_contribution, input_batch + transformed

        else:
            for j in range(self.num_condition_series):
                input_cut = tf.shape(input_batch[j])[1] - tf.shape(transformed)[1]
                condition = tf.slice(input_batch[j], [0, input_cut, 0], [-1, -1, -1])
                transformed = transformed + condition
            return skip_contribution, transformed


    def create_network(self, input_batch):
        if self.num_condition_series is None:
            self.variables = self.create_variables()
        else:
            self.variables = self.create_condition_vars()
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        initial_channels = 1
        with tf.name_scope('causal_layer'):
            current_layer = self._create_causal_layer(current_layer)

        if self.num_condition_series is None:
            output_width = tf.shape(input_batch)[1] - self.receptive_field + 1
        else:
            output_width = tf.shape(input_batch[0])[1] - self.receptive_field + 1


        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            #Linear layer
            w2 = self.variables['postprocessing']['postprocess']
            if self.use_biases:
                b2 = self.variables['postprocessing']['postprocess_bias']

            if self.histograms:
                tf.summary.histogram('postprocess_weights', w2)
                if self.use_biases:
                    tf.summary.histogram('postprocess_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            conv1 = tf.nn.conv1d(total, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b2)
            return conv1


    def time_series_to_sequences(self, time_series, log_difference, parallel=True):
        if self.num_condition_series is None:
            if parallel == True:
                num_sequences = int(np.shape(time_series)[0] / self.sequence_length) + 1
                num_leftover = num_sequences*self.sequence_length - time_series.shape[0]
                x = np.pad(time_series, ((0, num_leftover), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0)))
                sequences = np.reshape(x, (num_sequences, self.sequence_length, self.output_channels))
            else:
                if self.log_difference == True:
                    self.scaler = MinMaxScaler((1, np.max(time_series)))
                    time_series = self.scaler.fit_transform(time_series)
                    self.past = np.log(time_series[:-1])
                    time_series = np.log(time_series[1:]) - self.past
                sequences = []
                for i in range(time_series.shape[0]):
                    if i + self.receptive_field + self.forecast_horizon >= time_series.shape[0]:
                        break
                    past_samples = time_series[i:i + self.receptive_field]
                    future_samples = time_series[i + self.receptive_field + self.forecast_horizon - 1]
                    seq = np.vstack((past_samples, future_samples))
                    sequences.append(seq)
                sequences = np.array(sequences)
            return sequences
        else:
            sequences = []
            if parallel == True:
                for j in range(self.num_condition_series):
                    num_sequences = int(np.shape(time_series[:, j])[0] / self.sequence_length) + 1
                    num_leftover = num_sequences*self.sequence_length - time_series[:, j].shape[0]
                    x = np.pad(time_series[:, j], ((0, num_leftover)), mode='constant', constant_values=(0, 0))
                    sequences.append(np.reshape(x, (num_sequences, self.sequence_length, 1)))
                sequences = np.array(sequences)
            else:
                if self.log_difference == True:
                    self.scalers = []
                    for j in range(self.num_condition_series):
                        self.scalers.append(MinMaxScaler((1, np.max(time_series[:, j]+1))))
                        time_series[:, j] = self.scalers[j].fit_transform(time_series[:, j])
                    self.past = np.log(time_series[:-1, :])
                    time_series = np.log(time_series[1:, :]) - self.past
                for j in range(self.num_condition_series):
                    seqs = []
                    for i in range(time_series.shape[0]):
                        if i + self.receptive_field + self.forecast_horizon >= time_series.shape[0]:
                            break
                        past_samples = time_series[i:i + self.receptive_field, j]
                        future_samples = time_series[i + self.receptive_field + self.forecast_horizon - 1, j]
                        past_samples = past_samples.reshape(-1, 1)
                        seq = np.vstack((past_samples, future_samples))
                        seq = np.squeeze(seq)
                        seqs.append(seq)
                    seqs = np.array(seqs)
                    sequences.append(seqs)
                sequences = np.array(sequences)
                sequences = np.expand_dims(sequences, sequences.shape[-1])
            return sequences


    def loss(self, input_batch, l2_regularization_strength=False):
        # Cut off the last n samples of network input to preserve causality.
        if self.num_condition_series is None:
            batch_size = tf.shape(input_batch)[0]
            encoded = tf.reshape(input_batch, [batch_size, -1, self.output_channels])
            network_input = tf.reshape(input_batch, [batch_size, -1, self.output_channels])
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, -1])
            raw_output = self.create_network(network_input)
        else:
            batch_size = tf.shape(input_batch)[1]
            encoded = tf.reshape(input_batch[0, :, :, :], [batch_size, -1, 1])
            condition_input = tf.reshape(input_batch, [self.num_condition_series, batch_size, -1, 1])
            network_input_width = tf.shape(condition_input)[2] - 1
            network_input = tf.slice(condition_input, [0, 0, 0, 0], [-1, -1, network_input_width, -1])
            raw_output = self.create_network(network_input)

        with tf.name_scope('loss'):
            # Cut off the samples corresponding to the receptive field
            # for the first predicted sample.
            target_output = tf.slice(tf.reshape(encoded, [batch_size, -1, self.output_channels]), [0, self.receptive_field, 0], [-1, -1, -1])
            target_output = tf.reshape(target_output, [-1, self.output_channels])
            prediction = tf.reshape(raw_output, [-1, self.output_channels])

            #Mean Absolte Error
            loss = tf.reduce_mean(tf.abs(target_output - prediction))
            MASE = loss / tf.reduce_mean(tf.abs(encoded[:, 1:, :] - encoded[:, :-1, :]))
            tf.add_to_collection('prediction', prediction)
            tf.add_to_collection('target_output', target_output)
            tf.add_to_collection('MAE', loss)
            tf.add_to_collection('MASE', MASE)

            tf.summary.scalar('MAE', loss)
            tf.summary.scalar('MASE', MASE)

            if l2_regularization_strength == False:
                return loss, MASE, target_output, prediction
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (loss + l2_regularization_strength * l2_loss)

                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', total_loss)

                return total_loss, MASE, target_output, prediction


    def restore_model(self, tf_session, global_step):
        new_saver = tf.train.import_meta_graph(self.save_path + '-' + str(global_step-1) + '.meta')
        new_saver.restore(tf_session, tf.train.latest_checkpoint(self.log_dir))
        self.prediction = tf.get_collection('prediction')
        self.target_output = tf.get_collection('target_output')
        self.MAE = tf.get_collection('MAE')
        self.MASE = tf.get_collection('MASE')


    def inference_batches_to_series(self, tf_session, 
                                            sequences, 
                                            batch_index, 
                                            batch_size, 
                                            num_condition_series, 
                                            prediction, 
                                            target_output, 
                                            train_step = None, 
                                            place_holder=None,
                                            merged = None,
                                            writer = None,
                                            epoch = None):
        predicted = []
        actual = []
        for i in np.arange(0, sequences.shape[batch_index], batch_size):
            if num_condition_series is None:
                if place_holder is None:
                    inf_feed = {'input_sequence:0': sequences[i:i + batch_size, :, :]}
                else:
                    inf_feed = {place_holder: sequences[i:i + batch_size, :, :]}
            else:
                if place_holder is None:
                    inf_feed = {'condition_sequences:0': sequences[:, i:i + batch_size, :, :]}
                else:
                    inf_feed = {place_holder: sequences[:, i:i + batch_size, :, :]}
            if train_step is not None:
                tf_session.run(train_step, feed_dict=inf_feed)
            if merged is not None and writer is not None:
                if i % 1000 == 0:
                    summary = tf_session.run(merged, feed_dict=inf_feed)
                    writer.add_summary(summary, int(epoch*sequences.shape[batch_index]/batch_size) + i)
            p, a = tf_session.run([prediction, target_output], feed_dict=inf_feed)
            if i==0:
                predicted = p[0]
                actual = a[0]
            else:
                predicted = np.vstack((predicted, p[0]))
                actual = np.vstack((actual, a[0]))
        return predicted, actual

    def train(self, time_series, batch_size, log_difference, epochs, train_fraction=0.8):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.log_difference = log_difference
        self.set_parameters(time_series)

        tf_session = tf.Session()
        x_ = self.create_placeholders()
        self.MAE, self.MASE, self.target_output, self.prediction = self.loss(x_)
        tf.set_random_seed(self.random_seed)
        merged = tf.summary.merge_all()
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.MAE)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)
        tf_saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(self.train_log, tf_session.graph)
        val_writer = tf.summary.FileWriter(self.val_log, tf_session.graph)
        sequences = self.time_series_to_sequences(time_series, self.log_difference, parallel=False)
        print 'input shape:', sequences.shape
        num_sequences = sequences.shape[self.batch_index]
        train_stop = int(self.train_fraction*num_sequences)

        if self.num_condition_series is None:
            train_x = sequences[0:train_stop]
            val_x = sequences[train_stop:]
        else: 
            train_x = sequences[:, 0:train_stop, :, :]
            val_x = sequences[:, train_stop:, :, :]

        for e in range(epochs):
            print 'step{}:'.format(e)
            predicted_train, actual_train = self.inference_batches_to_series(tf_session, 
                                                                            train_x, 
                                                                            self.batch_index, 
                                                                            self.batch_size, 
                                                                            self.num_condition_series, 
                                                                            self.prediction, 
                                                                            self.target_output, 
                                                                            train_step = train_step, 
                                                                            place_holder=x_, 
                                                                            merged=merged,
                                                                            writer=train_writer,
                                                                            epoch=e)
            predicted_val, actual_val = self.inference_batches_to_series(tf_session, 
                                                                        val_x, 
                                                                        self.batch_index, 
                                                                        self.batch_size, 
                                                                        self.num_condition_series, 
                                                                        self.prediction, 
                                                                        self.target_output, 
                                                                        train_step = None, 
                                                                        place_holder=x_,
                                                                        merged=merged,
                                                                        writer=val_writer,
                                                                        epoch=e)
            
            mae = np.mean(np.abs(predicted_train - actual_train))
            trivial = np.mean(np.abs(actual_train[1:] - actual_train[:-1]))
            print 'Train MAE:', mae
            print 'Train MASE:', mae/trivial

            mae = np.mean(np.abs(predicted_val - actual_val))
            trivial = np.mean(np.abs(actual_val[1:] - actual_val[:-1]))
            print 'Val MAE:', mae
            print 'Val MASE:', mae/trivial
            tf_saver.save(tf_session, self.save_path, global_step=e)



    def predict_one_time_step(self, time_series, batch_size, log_difference, global_step=10, tf_session=None):
        self.batch_size = batch_size
        self.log_difference = log_difference
        self.set_parameters(time_series)
        sequences = self.time_series_to_sequences(time_series, self.log_difference, parallel=False)
        if tf_session is None:
            tf_session = tf.Session()
            self.restore_model(tf_session, global_step)
        predicted, actual = self.inference_batches_to_series(tf_session, 
                                                                sequences, 
                                                                self.batch_index, 
                                                                self.batch_size, 
                                                                self.num_condition_series, 
                                                                self.prediction, 
                                                                self.target_output, 
                                                                train_step = None, 
                                                                place_holder=None,
                                                                merged = None,
                                                                writer = None,
                                                                epoch=None)


        if self.log_difference == True:
            predicted, actual = self.inverse_transform(predicted, actual)
        mae = np.mean(np.abs(predicted - actual))
        trivial = np.mean(np.abs(actual[self.forecast_horizon:] - actual[:-self.forecast_horizon]))
        print 'Test MAE:', mae
        print 'Test MASE:', mae/trivial
        
        return predicted, actual


    def predict_n_time_steps(self, time_series, n, batch_size, log_difference, global_step=10, tf_session=None):
        self.batch_size = batch_size
        self.log_difference = self.log_difference
        self.set_parameters(time_series)
        sequences = self.time_series_to_sequences(time_series, self.log_difference, parallel=False)
        if tf_session is None:
            tf_session = tf.Session()
            self.restore_model(tf_session, global_step)
        predicted = []
        actual = []
        for i in np.arange(0, sequences.shape[self.batch_index], self.batch_size):
            if self.num_condition_series is None:
                current_batch = sequences[i:i + self.batch_size, :, :]
                inf_feed = {'input_sequence:0': current_batch}
            else:
                current_batch = sequences[:, i:i + self.batch_size, :, :]
                inf_feed = {'condition_sequences:0': current_batch}
            p, a = tf_session.run([self.prediction, self.target_output], inf_feed)
            rolled = current_batch
            for j in range(n-1):
                if self.num_condition_series is None:
                    rolled = np.roll(rolled, shift=rolled.shape[1]-1, axis=1)
                    rolled[:, -1, :] = p[0]
                    feed = {'input_sequence:0': rolled}
                else:
                    rolled = np.roll(rolled, shift=rolled.shape[2]-1, axis=2)
                    rolled[0, :, -1, :] = p[0]
                    feed = {'condition_sequences:0': rolled}
                p = tf_session.run(self.prediction, feed)
            if i==0:
                predicted = p[0]
                actual = a[0]
            else:
                predicted = np.vstack((predicted, p[0]))
                actual = np.vstack((actual, a[0]))

        if self.log_difference == True:
            predicted, actual = self.inverse_transform(predicted, actual)

        mae = np.mean(np.abs(predicted[:-n] - actual[n:]))
        mase = mae/np.mean(np.abs(predicted[:-n] - predicted[n:]))
        print 'Test MAE:', mae
        print 'Test MASE:', mase
        return predicted, actual


    def plot_predicted_vs_actual(self, time_series, n, batch_size, log_difference, global_step=10):
        if n > 1:
            predicted, actual = self.predict_n_time_steps(time_series, n, batch_size, log_difference, global_step)
            plt.plot(predicted, label='predicted', color='b')
            plt.plot(actual, label='actual', color='r')
            plt.legend()
            plt.show()
        else:
            predicted, actual = self.predict_one_time_step(time_series, batch_size, log_difference, global_step)
            plt.plot(predicted, label='predicted', color='b')
            plt.plot(actual, label='actual', color='r')
            plt.legend()
            plt.show()





if __name__ == '__main__':
    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    node0 = ercot.all_nodes[0]
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[100])
    train, test = ercot.get_train_test(node0, normalize=False, include_seasonal_vectors=False)
    wavenet = WaveNet(MIMO=False, forecast_horizon=1)
    wavenet.train(train, batch_size=128, log_difference=True, epochs=10)
    predicted1, actual1 = wavenet.predict_one_time_step(test, batch_size=128, log_difference=True, global_step=9)
    wavenet.delete_dirs()



    