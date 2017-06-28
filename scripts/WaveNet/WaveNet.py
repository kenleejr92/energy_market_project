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


def time_series_to_batches(time_series, sequence_length, batch_size):
    if sequence_length == -1:
        x = np.reshape(time_series, (1, time_series.shape[0], 1))
        return x
    else:
        num_sequences = int(np.shape(time_series)[0] / sequence_length) + 1
        num_leftover = num_sequences*sequence_length - time_series.shape[0]
        x = np.pad(time_series, ((0, num_leftover)), mode='constant', constant_values=(0, 0))
        x = np.reshape(x, (num_sequences, sequence_length, 1))

        num_batches = int(num_sequences / batch_size) + 1
        num_leftover = num_batches*batch_size - num_sequences
        x = np.pad(x, ((0, num_leftover), (0, 0), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        batches = np.reshape(x, (num_batches, batch_size, sequence_length, 1))
        return batches


class WaveNet(object):

    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.filter_width = 2
        self.residual_channels = 32
        self.dilation_channels = 32
        self.output_channels = 1
        self.skip_channels = 256
        self.use_biases = True
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        self.variables = self.create_variables()
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.histograms = True


    def calculate_receptive_field(self, filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + filter_width
        print receptive_field
        return receptive_field


    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_ = tf.placeholder('float', [None, self.sequence_length, 1], name='input_sequence')

        return x_

    def create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''
        # scalar (univariate) prediction
        initial_channels = 1
        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                layer['filter'] = weight_variable([self.filter_width, initial_channels, self.residual_channels], 'filter')
                var['causal_layer'] = layer

        var['dilated_stack'] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = weight_variable([self.filter_width, self.residual_channels, self.dilation_channels], 'filter')
                    current['gate'] = weight_variable([self.filter_width, self.residual_channels, self.dilation_channels], 'gate')
                    current['dense'] = weight_variable([1, self.dilation_channels, self.residual_channels], 'dense')
                    current['skip'] = weight_variable([1, self.dilation_channels, self.skip_channels], 'skip')


                    if self.use_biases:
                        current['filter_bias'] = bias_variable([self.dilation_channels], 'filter_bias')
                        current['gate_bias'] = bias_variable([self.dilation_channels], 'gate_bias')
                        current['dense_bias'] = bias_variable([self.residual_channels], 'dense_bias')
                        current['skip_bias'] = bias_variable([self.skip_channels], 'slip_bias')

                    var['dilated_stack'].append(current)

        with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = weight_variable([1, self.skip_channels, self.skip_channels], 'postprocess1')
                current['postprocess2'] = weight_variable([1, self.skip_channels, self.output_channels], 'postprocess2')
                if self.use_biases:
                    current['postprocess1_bias'] = bias_variable([self.skip_channels], 'postprocess1_bias')
                    current['postprocess2_bias'] = bias_variable([self.output_channels], 'postprocess2_bias')
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)


    def _create_dilation_layer(self, input_batch, layer_index, dilation, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> residual output
               |------------------------------------|

        sum(skip_outputs) -> 1x1 conv --> output

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        # out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

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
            tf.summary.histogram(layer + '_gate', weights_gate)
            tf.summary.histogram(layer + '_dense', weights_dense)
            tf.summary.histogram(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.summary.histogram(layer + '_biases_filter', filter_bias)
                tf.summary.histogram(layer + '_biases_gate', gate_bias)
                tf.summary.histogram(layer + '_biases_dense', dense_bias)
                tf.summary.histogram(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def create_network(self, input_batch):
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        initial_channels = 1

        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1


        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.summary.histogram('postprocess1_weights', w1)
                tf.summary.histogram('postprocess2_weights', w2)
                if self.use_biases:
                    tf.summary.histogram('postprocess1_biases', b1)
                    tf.summary.histogram('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            # transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(total, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b2)
            # transformed2 = tf.nn.relu(conv1)
            # conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            # if self.use_biases:
            #     conv2 = tf.add(conv2, b2)
            return conv1


    def time_series_to_batches(self, time_series):
        if self.sequence_length == -1:
            x = np.reshape(time_series, (1, time_series.shape[0], 1))
            return x
        else:
            num_sequences = int(np.shape(time_series)[0] / self.sequence_length) + 1
            num_leftover = num_sequences*self.sequence_length - time_series.shape[0]
            x = np.pad(time_series, ((0, num_leftover)), mode='constant', constant_values=(0, 0))
            batches = np.reshape(x, (num_sequences, self.sequence_length, 1))
            return batches


    def loss(self, input_batch, l2_regularization_strength = False):
        # Cut off the last sample of network input to preserve causality.
        batch_size = tf.shape(input_batch)[0]
        encoded = tf.reshape(input_batch, [batch_size, -1, 1])
        network_input = tf.reshape(input_batch, [batch_size, -1, 1])
        network_input_width = tf.shape(network_input)[1] - 1
        network_input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, -1])
        
        raw_output = self.create_network(network_input)

        with tf.name_scope('loss'):
            # Cut off the samples corresponding to the receptive field
            # for the first predicted sample.
            #subtract 1 from receptive field??????????????????
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
                return loss, target_output, prediction, raw_output
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (loss + l2_regularization_strength * l2_loss)

                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', total_loss)

                return total_loss


    def restore_model(self, tf_session, global_step=29):
        new_saver = tf.train.import_meta_graph(SAVE_PATH + '-' + str(global_step) + '.meta')
        new_saver.restore(tf_session, tf.train.latest_checkpoint(LOG_DIR))
        self.prediction = tf.get_collection('prediction')
        self.target_output = tf.get_collection('target_output')
        self.MAE = tf.get_collection('MAE')


    def train(self, time_series, epochs=30):

        tf_session = tf.Session()
        x_ = self.create_placeholders()
        self.MAE, self.target_output, self.prediction, self.raw_output = self.loss(x_)
        merged = tf.summary.merge_all()
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.MAE)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)
        tf_saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(TRAIN_LOG, tf_session.graph)


        batches = self.time_series_to_batches(time_series)
        for e in range(epochs):
            for i in np.arange(0, batches.shape[0], self.batch_size):
                tr_feed = {x_: batches[i:i+self.batch_size, :, :]}
                tf_session.run(train_step, feed_dict=tr_feed)
                
            tr_feed = {x_: batches[0:0+self.batch_size, :, :]}
            print tf_session.run(self.MAE, feed_dict=tr_feed)
            predicted, actual, raw_output = tf_session.run([self.prediction, self.target_output, self.raw_output], feed_dict=tr_feed)
            tf_saver.save(tf_session, SAVE_PATH, global_step=e)
            summary_train = tf_session.run(merged, feed_dict=tr_feed)
            train_writer.add_summary(summary_train, e)


    def inference(self, time_series):
        ts = self.time_series_to_batches(time_series)

        tf_session = tf.Session()
        self.restore_model(tf_session)
        inf_feed = {'input_sequence:0': ts[0:self.batch_size]}
        predicted, actual, MAE = tf_session.run([self.prediction, self.target_output, self.MAE], inf_feed)
        print MAE
        plt.plot(predicted[0][0:100], label='predicted')
        plt.plot(actual[0][0:100], label='actual')
        plt.legend()
        plt.show()




if __name__ == '__main__':
    sine_wave = np.sin(np.arange(0, 20000, 0.5))
    wavenet = WaveNet(batch_size=128, sequence_length=8000)
    wavenet.inference(sine_wave)
    # tf_session = tf.Session()
    # wavenet = WaveNet()
    # x_, y_ = wavenet.create_placeholders()
    # output = wavenet.create_network(x_)
    # init_op = tf.global_variables_initializer()
    # tf_session.run(init_op)
    # print tf_session.run(tf.shape(output), feed_dict={x_: test})


    # ercot = ercot_data_interface()
    # sources_sinks = ercot.get_sources_sinks()
    # nn = ercot.get_nearest_CRR_neighbors(sources_sinks[5])
    # train, test, val = ercot.get_train_test_val(nn[0])