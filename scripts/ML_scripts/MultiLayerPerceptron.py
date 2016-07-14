
# coding: utf-8

from __future__ import print_function

__docformat__ = 'restructedtext en'

import sys
import numpy
import theano
import theano.tensor as T
sys.path.insert(0, '/home/kenlee/energy_market_project/scripts/MySQL_scripts/')

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


    def errors(self, y, output_norm):
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
            MAE = T.mean(abs(y*output_norm-self.y_pred.T*output_norm))
            MAPE = T.mean(abs((y*output_norm-self.y_pred.T*output_norm)/(y*output_norm)))*100
            TheilU1 = T.sqrt(T.mean((y*output_norm-self.y_pred.T*output_norm)**2))/(T.sqrt(T.mean((y*output_norm)**2))+T.sqrt(T.mean((self.y_pred*output_norm)**2)))
            return MAE, MAPE, TheilU1
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer = OutputLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # quadratic cost of MLP
        self.cost_function = (
            self.outputLayer.cost_function
        )
        # same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

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









