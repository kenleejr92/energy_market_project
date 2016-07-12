__author__ = 'kenlee'

import sys
import os
import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as T
from MultiLayerPerceptron import MLP
from DAM_prices_by_SP import Query_DAM_by_SP
from DAM_prices_by_SP import train_test_validate

def load_data(start_date, end_date, zone, model):

    #############
    # LOAD DATA #
    #############


    # Load the dataset
    qdsp = Query_DAM_by_SP()
    qdsp.query(start_date, end_date)
    feature_targets = qdsp.construct_feature_vector_matrix(zone, model)
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

def test_mlp(f,
             start_date,
             end_date,
             zone,
             model,
             learning_rate=0.01,
             L1_reg=0.00,
             L2_reg=0.0001,
             n_hidden=30,):
    """

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(start_date, end_date, zone, model)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    x = T.fmatrix('x')  # the data is presented as rasterized images
    y = T.fmatrix('y') #  the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    predictor = MLP(
        rng=rng,
        input=x,
        n_in=train_set_x.eval().shape[1],
        n_hidden=n_hidden,
        n_out=1
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (predictor.cost_function(y) + L1_reg * predictor.L1 + L2_reg * predictor.L2_sqr)

    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[],
        outputs=predictor.errors(y),
        givens={
            x: test_set_x,
            y: test_set_y
        }
    )

    validate_model = theano.function(
        inputs=[],
        outputs=predictor.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in predictor.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(predictor.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )

    mlp_prediction = theano.function(
        inputs=[],
        outputs=predictor.outputLayer.y_pred,
        givens = {
            x: test_set_x,
        }
    )

    val_errors = []
    test_errors = []
    training_epochs = numpy.arange(2000, 4500, 250)
    for epoch in range(4000):
        cost = train_model()
        if epoch in training_epochs:
            MAE_val, MAPE_val, TheilU1_val = validate_model()
            val_errors.append((float(MAE_val), float(MAPE_val), float(TheilU1_val)))
            MAE_test, MAPE_test, TheilU1_test = test_model()
            test_errors.append((float(MAE_test), float(MAPE_test), float(TheilU1_test)))
    val_metric = [val_errors[i][1] for i in range(len(val_errors))]
    min_val_idx = val_metric.index(min(val_metric))
    optimal_epoch = training_epochs[min_val_idx]
    f.write('%s,%s,%f,%f,%f,%d\n' % (zone,
                                 start_date[:4],
                                 test_errors[min_val_idx][0],
                                 test_errors[min_val_idx][1],
                                 test_errors[min_val_idx][2],
                                 optimal_epoch))

    # print('Minimum MAPE on val_set for %d optimal epochs: %f' % (optimal_epoch, min(val_metric)))
    # print('MAE on test_set: %f' % test_errors[min_val_idx][0])
    # print('MAPE on test_set: %f' % test_errors[min_val_idx][1])
    # print('TheilU1 on test_set: %f' % test_errors[min_val_idx][2])
    # mlp_pred = mlp_prediction()
    # i = numpy.arange(mlp_pred.shape[0])
    # plt.plot(i, test_set_y.T.eval(), label='Actual Price')
    # plt.plot(i, mlp_pred, label='Predicted Price')
    # plt.xlabel('Hour')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()
DATES = [('2011-01-01', '2011-12-31'),
         ('2012-01-01', '2012-12-31'),
         ('2013-01-01', '2013-12-31'),
         ('2014-01-01', '2014-12-31'),
         ('2015-01-01', '2015-12-31')]
LOAD_ZONES = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']

if __name__ == '__main__':
    model = 'A'
    os.chdir('../test_results')
    f = open('MLP_Model%s_results.csv' % model, 'w+')
    f.write('zone,year,MAE,MAPE,TheilU1,epochs\n')
    for sd, ed in DATES:
        for sp in LOAD_ZONES:
            test_mlp(f, sd, ed, sp, model)
        print('Finished year %s' % sd[:4])
    f.close()
