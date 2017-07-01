import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from ercot_data_interface import ercot_data_interface
import matplotlib.pyplot as plt
from ops import * 


LOG_DIR = '/home/kenleejr92/energy_market_project/scripts/WaveNet/tmp'
SAVE_PATH = LOG_DIR + '/WaveNet'
TRAIN_LOG = LOG_DIR + '/train'
VAL_LOG = LOG_DIR + '/val'



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
    def __init__(self, sequence_length, output_channels=1, num_condition_series=None):
        self.train_fraction = 0.8
        self.sequence_length = sequence_length
        self.initial_filter_width = 10
        self.filter_width = 2
        self.residual_channels = 32
        self.dilation_channels = 32
        self.output_channels = 1
        self.skip_channels = 256
        self.use_biases = True
        self.use_batch_norm = False
        self.dilations = [1, 2, 4, 8]
        self.num_condition_series = num_condition_series
        if self.num_condition_series is None:
            self.variables = self.create_variables()
        else:
            self.variables = self.create_condition_vars()
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.histograms = True


    def calculate_receptive_field(self, filter_width, dilations):
        '''
        Final receptive field for model
        '''
        receptive_field = (filter_width - 1) * sum(dilations) + self.initial_filter_width
        print receptive_field
        return receptive_field


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
                        current_receptive_field = 1 + self.initial_filter_width
                        current['filter_scale'] = scale_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_scaler')
                        current['filter_offset'] = offset_variable([self.sequence_length - current_receptive_field, self.residual_channels], 'BN_offset')

                    var['condition_layer'].append(current)

        var['dilated_stack'] = list()
        with tf.variable_scope('dilation_stack_par'):        
            for i, dilation in enumerate(self.dilations):
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
                        current_receptive_field = np.sum(self.dilations[:i]) + 1 + self.initial_filter_width
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
        initial_channels = 1
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
                        current_receptive_field = np.sum(self.dilations[:i]) + 1 + self.initial_filter_width
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


    def time_series_to_batches(self, time_series):
        #TO DO: reshape for output channels
        if self.num_condition_series is None:
            if self.output_channels > 1:
                pass
            num_sequences = int(np.shape(time_series)[0] / self.sequence_length) + 1
            num_leftover = num_sequences*self.sequence_length - time_series.shape[0]
            x = np.pad(time_series, ((0, num_leftover)), mode='constant', constant_values=(0, 0))
            batches = np.reshape(x, (num_sequences, self.sequence_length, 1))
            return batches
        else:
            batches = []
            for j in range(self.num_condition_series):
                num_sequences = int(np.shape(time_series[:, j])[0] / self.sequence_length) + 1
                num_leftover = num_sequences*self.sequence_length - time_series[:, j].shape[0]
                x = np.pad(time_series[:, j], ((0, num_leftover)), mode='constant', constant_values=(0, 0))
                batches.append(np.reshape(x, (num_sequences, self.sequence_length, 1)))
            return np.array(batches)


    def loss(self, input_batch, l2_regularization_strength=0.01):
        # Cut off the last sample of network input to preserve causality.
        if self.num_condition_series is None:
            batch_size = tf.shape(input_batch)[0]
            encoded = tf.reshape(input_batch, [batch_size, -1, 1])
            network_input = tf.reshape(input_batch, [batch_size, -1, 1])
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
            tf.add_to_collection('prediction', prediction)
            tf.add_to_collection('target_output', target_output)
            tf.add_to_collection('MAE', loss)

            tf.summary.scalar('loss', loss)

            if l2_regularization_strength == False:
                return loss, target_output, prediction
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (loss + l2_regularization_strength * l2_loss)

                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', total_loss)

                return total_loss, target_output, prediction


    def restore_model(self, tf_session, global_step=99):
        new_saver = tf.train.import_meta_graph(SAVE_PATH + '-' + str(global_step) + '.meta')
        new_saver.restore(tf_session, tf.train.latest_checkpoint(LOG_DIR))
        self.prediction = tf.get_collection('prediction')
        self.target_output = tf.get_collection('target_output')
        self.MAE = tf.get_collection('MAE')


    def train(self, time_series, batch_size, epochs=100):

        tf_session = tf.Session()
        x_ = self.create_placeholders()
        self.MAE, self.target_output, self.prediction = self.loss(x_)
        merged = tf.summary.merge_all()
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.MAE)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)
        tf_saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(TRAIN_LOG, tf_session.graph)
        val_writer = tf.summary.FileWriter(VAL_LOG, tf_session.graph)

        batches = self.time_series_to_batches(time_series)

        if self.num_condition_series is None:
            num_batches = batches.shape[0]
            train_stop = int(self.train_fraction*num_batches)
            train_x = batches[0:train_stop]
            val_x = batches[train_stop:]
            for e in range(epochs):
                for i in np.arange(0, train_x.shape[0], batch_size):
                    tr_feed = {x_: train_x[i:i + batch_size, :, :]}
                    tf_session.run(train_step, feed_dict=tr_feed)

                tr_feed = {x_: train_x}
                MAE = tf_session.run(self.MAE, feed_dict=tr_feed)
                print 'Train MAE:', MAE
                predicted, actual = tf_session.run([self.prediction, self.target_output], feed_dict=tr_feed)
                MASE = MAE/np.mean(np.abs(actual[1:] - actual[:-1]))
                print 'Train MASE:', MASE
                summary_train = tf_session.run(merged, feed_dict=tr_feed)
                train_writer.add_summary(summary_train, e)

                val_feed = {x_: val_x}
                MAE = tf_session.run(self.MAE, feed_dict=val_feed)
                print 'Validation MAE:', MAE
                predicted, actual = tf_session.run([self.prediction, self.target_output], feed_dict=val_feed)
                MASE = MAE/np.mean(np.abs(actual[1:] - actual[:-1]))
                print 'VAL MASE:', MASE
                summary_val = tf_session.run(merged, feed_dict=val_feed)
                val_writer.add_summary(summary_val, e)
                if e==99:
                    plt.plot(predicted, label='predicted')
                    plt.plot(actual, label='actual')
                    plt.legend()
                    plt.show()
                tf_saver.save(tf_session, SAVE_PATH, global_step=e)
        else:
            num_batches = batches.shape[1]
            train_stop = int(self.train_fraction*num_batches)
            train_x = batches[:, 0:train_stop, :, :]
            val_x = batches[:, train_stop:, :, :]
            for e in range(epochs):
                for i in np.arange(0, train_x.shape[1], batch_size):
                    tr_feed = {x_:train_x[:, i:i + batch_size, :, :]}
                    tf_session.run(train_step, feed_dict=tr_feed)
                    
                tr_feed = {x_: train_x}
                MAE = tf_session.run(self.MAE, feed_dict=tr_feed)
                print 'Train MAE:', MAE
                predicted, actual = tf_session.run([self.prediction, self.target_output], feed_dict=tr_feed)
                MASE = MAE/np.mean(np.abs(actual[1:] - actual[:-1]))
                print 'Train MASE:', MASE
                summary_train = tf_session.run(merged, feed_dict=tr_feed)
                train_writer.add_summary(summary_train, e)

                val_feed = {x_: val_x}
                MAE = tf_session.run(self.MAE, feed_dict=val_feed)
                print 'Validation MAE:', MAE
                predicted, actual = tf_session.run([self.prediction, self.target_output], feed_dict=val_feed)
                print 'VAL MASE:', MASE
                summary_val = tf_session.run(merged, feed_dict=val_feed)
                summary_val = tf_session.run(merged, feed_dict=val_feed)
                val_writer.add_summary(summary_val, e)

                if e==99:
                    plt.plot(predicted, label='predicted')
                    plt.plot(actual, label='actual')
                    plt.legend()
                    plt.show()
                tf_saver.save(tf_session, SAVE_PATH, global_step=e)


    def inference(self, time_series):
        batches = self.time_series_to_batches(time_series)
        tf_session = tf.Session()
        self.restore_model(tf_session)
        if self.num_condition_series is None:
            inf_feed = {'input_sequence:0': batches}
            predicted, actual, MAE = tf_session.run([self.prediction, self.target_output, self.MAE], inf_feed)
            print MAE[0]
            plt.plot(predicted[0], label='predicted')
            plt.plot(actual[0], label='actual')
            plt.legend()
            plt.show()
        else:
            inf_feed = {'condition_sequences:0': batches}
            predicted, actual, MAE = tf_session.run([self.prediction, self.target_output, self.MAE], inf_feed)
            print MAE[0]
            plt.plot(predicted[0], label='predicted')
            plt.plot(actual[0], label='actual')
            plt.legend()
            plt.show()

def time_series_to_batches(time_series, num_condition_series, sequence_length):
        if num_condition_series is None:
            num_sequences = int(np.shape(time_series)[0] / sequence_length) + 1
            num_leftover = num_sequences*sequence_length - time_series.shape[0]
            x = np.pad(time_series, ((0, num_leftover)), mode='constant', constant_values=(0, 0))
            batches = np.reshape(x, (num_sequences, sequence_length, 1))
            return batches
        else:
            batches = []
            for j in range(num_condition_series):
                num_sequences = int(np.shape(time_series[:, j])[0] / sequence_length) + 1
                num_leftover = num_sequences*sequence_length - time_series[:, j].shape[0]
                x = np.pad(time_series[:, j], ((0, num_leftover)), mode='constant', constant_values=(0, 0))
                batches.append(np.reshape(x, (num_sequences, sequence_length, 1)))
            return np.array(batches)


if __name__ == '__main__':
    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[5])
    train, test = ercot.get_train_test(nn[0])
    # prices3 = np.squeeze(ercot.query_prices(nn[0], '2011-01-01', '2015-01-01').as_matrix())
    # prices4 = np.squeeze(ercot.query_prices(nn[0], '2015-01-01', '2016-5-23').as_matrix())
    print train.shape
    wavenet = WaveNet(sequence_length=48, num_condition_series=train.shape[1])
    wavenet.train(train, batch_size=256)
    wavenet.inference(test)
    