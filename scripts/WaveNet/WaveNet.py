import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from ercot_data_interface import ercot_data_interface

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

def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        #shape = [number of batches, number of samples, number of channels]
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

class WaveNet(object):

    def __init__(self):
        self.input_length = None
        self.initial_filter_width = 2
        self.residual_channels = 5
        self.variables = create_variables()



    def create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = 1
                initial_filter_width = self.initial_filter_width
                layer['filter'] = weight_variable([initial_filter_width, initial_channels, self.residual_channels])
                var['causal_layer'] = layer


        return var


if __name__ == '__main__':
    tf_session = tf.Session()
    x = np.arange(1,100)
    x = np.expand_dims(x, 1)
    x = np.expand_dims(x, 0)
    print x.shape
    y = time_to_batch(x, dilation=2)
    print tf_session.run(tf.shape(y))