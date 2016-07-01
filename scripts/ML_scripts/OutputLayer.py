
# coding: utf-8

# In[10]:

"""
Output Layer for an MLP used for regression
"""

from __future__ import print_function

import numpy

import theano
import theano.tensor as T
import sys
sys.path.insert(0, '/home/kenlee/energy_market_project/scripts/MySQL_scripts/')
from DAM_prices_by_SP import Query_DAM_by_SP
from DAM_prices_by_SP import train_test_validate


class OutputLayer(object):
    """
    For regression, output layer of MLP is just a linear sum of the activations times the weights
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the linear regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # output of MLP is dot product of input with weights of final layer plus the bias
        self.y_pred = self.p_y_given_x = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def cost_function(self, y):
        # quadratic cost function for regression
        return T.mean((y - self.y_pred.T)**2)
    

    def errors(self, y):
        """Return a float representing mean quadratic error over the minibatch 

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean((y-self.y_pred.T)**2)
        else:
            raise NotImplementedError()

def load_data():

    #############
    # LOAD DATA #
    #############

    
    # Load the dataset
    qdsp = Query_DAM_by_SP()
    qdsp.query("2015-01-01","2015-12-31")
    feature_targets = qdsp.construct_feature_vector_matrix("HB_BUSAVG","A")
    train_set, val_set, test_set = train_test_validate(feature_targets)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 2 dimensions that has the same
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asmatrix(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asmatrix(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(val_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval

def test_linreg():
    # Testing Linear Regression
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #####################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.fmatrix('x')  # data, presented as rasterized images
    y = T.fmatrix('y')  # labels, presented as 1D vector of [int] labels

    # construct the linear regression class
    # feature vectors are 57 units wide
    linreg = OutputLayer(input=x, n_in=57, n_out=1)

    # the cost we minimize during training is the squared error cost
    # the model in symbolic format
    cost = linreg.cost_function(y)
    prediction = linreg.y_pred
    # compiling a Theano function that computes the mistakes that are made by
    # the model on the test, validation, and training sets
    test_model = theano.function(
        inputs=[index],
        outputs=linreg.errors(y),
        givens={
            x: test_set_x,
            y: test_set_y
        },
        on_unused_input='ignore'
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=linreg.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        },
        on_unused_input='ignore'
    )
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=linreg.W)
    g_b = T.grad(cost=cost, wrt=linreg.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    learning_rate = 0.13
    updates = [(linreg.W, linreg.W - learning_rate * g_W),
               (linreg.b, linreg.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x,
            y: train_set_y
        },
        on_unused_input='ignore'
    )

    test_theano = theano.function(
        inputs=[index],
        outputs = cost,
        givens={
            x: train_set_x,
            y: train_set_y
        },
        on_unused_input='ignore'
    )

    pred = train_model(0)
    print(pred)
    print(linreg.W.eval())
    print(linreg.b.eval())
