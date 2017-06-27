import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from ercot_data_interface import ercot_data_interface
import matplotlib.pyplot as plt
from ops import * 

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

    def __init__(self):
        self.input_length = None
        self.filter_width = 2
        self.sequence_length = 168
        self.residual_channels = 5
        self.dilation_channels = 5
        self.output_channels = 1
        self.skip_channels = 5
        self.use_biases = True
        self.dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.variables = self.create_variables()
        self.receptive_field = self.calculate_receptive_field(self.filter_width, self.dilations)
        self.histograms = True


    def calculate_receptive_field(self, filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1 + filter_width - 1
        return receptive_field

    def preprocess_time_series(self, time_series, batch_size):
        if batch_size == 1:
            y = np.append(time_series, time_series[-1])[1:]
            x = time_series
            return x, y[self.receptive_field-1:]
        else:
            np.reshape(time_series, (batch_size, time_series.shape[0]))


    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_ = tf.placeholder('float', [None, self.sequence_length, 1], name='input_sequence')
        y_ = tf.placeholder('float', [None, self.sequence_length, 1], name='target_sequence')

        return x_, y_

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

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
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

    def loss(self):
        pass



    def train(self):
        pass


if __name__ == '__main__':
    test = np.arange(0, 168)
    
    tf_session = tf.Session()
    wavenet = WaveNet()
    x, y = wavenet.preprocess_time_series(test, batch_size=1)
    plt.plot(x)
    plt.plot(y)
    plt.show()
    test = np.expand_dims(test, 1)
    test = np.expand_dims(test, 0)
    print x.shape, y.shape
    x_, y_ = wavenet.create_placeholders()
    output = wavenet.create_network(x_)
    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)
    print wavenet.receptive_field
    print test[:, wavenet.receptive_field-1:].shape
    print tf_session.run(tf.shape(output), feed_dict={x_: test})


    ercot = ercot_data_interface()
    sources_sinks = ercot.get_sources_sinks()
    nn = ercot.get_nearest_CRR_neighbors(sources_sinks[5])
    train, test, val = ercot.get_train_test_val(nn[0])
    x, y = wavenet.preprocess_time_series(train, batch_size=1)
    print x.shape, y.shape