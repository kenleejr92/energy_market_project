import numpy as np
from WaveNet import WaveNet
from MLP import MLP
from ARIMA import ARIMA
from ercot_data_interface import ercot_data_interface



if __name__ == '__main__':
    #Get a train and test set
    ercot = ercot_data_interface()
    #sources_sinks are the names of nodes in CRRs
    sources_sinks = ercot.get_sources_sinks()
    try:
        #nn are the nearest neighbors of a given source_sink
        nn = ercot.get_nearest_CRR_neighbors(sources_sinks[150])
        #train and test are matrices of dimension [num_samples, num_nearest_neighbors]
        #the first column of each matrix is the 'center node'
        train, test = ercot.get_train_test(nn, include_seasonal_vectors=False)
    except Exception as error:
        print(repr(error))

    #This gets the center node
    center_node_train = np.expand_dims(train[:, 0], 1)
    center_node_test = np.expand_dims(test[:, 0], 1)


    print 'ARIMA:'
    #"seasonal" was replaced with d
    arima = ARIMA(p=2, d=1, q=2, log_difference=False)
    arima.fit(center_node_train)
    predicted, actual = arima.predict(center_node_test)
    arima.print_statistics(predicted, actual)

    print 'MLP:'
    mlp = MLP(random_seed=1234, log_difference=False, forecast_horizon=1)
    mlp.train(center_node_train, look_back=48)
    predicted, actual = mlp.predict(center_node_test)
    mlp.print_statistics(predicted, actual)

    print 'WaveNet:'
    '''
    foreast horizon: how far into the future you want to predict (always 1 for now)
    log_difference: if True, predicts the log difference of the time series
    initial_filter_width: length of the first convolutional filter
    filter_width: length of dilation filters
    residual/dilation/skip channels: leave as 32, 32, 256
    use_biases: whether to use biases in network, leave True
    use_batch_norm: batch_normalization, this is a form of pre-processing learned by the network. we will test how this improves
                    or decreases the performance
    dilations: list of dilation factors
                Remember the "p" in ARIMA(p, d, q) is how far back in the past the algorithm looks to predict the future.
                The "p" for WaveNet is equal to filter_width*sum(dilation_factors) + initial_filter_width
                We want to make this on the order of 200 to capture the weekly dependencies. Also, make sure dilation_factors
                start from 1 and go up in powers of 2. 
    random_seed: random seed from which weights are drawn, ensures a consistent method of getting the same results each time
    MIMO: (multi-input multi-output) If you pass multiple time series into the network and MIMO is False, the network will adjust it's topoplogy
                                    to condition on the time series to predict the first time series in the list
                                    If you pass multiple time series into the network and MIMO is True, the network will adjust it's topoplogy
                                    to predict all time_series simultaneously
    '''
    wavenet = WaveNet(forecast_horizon=1, 
                        log_difference=False, 
                        initial_filter_width=2, 
                        filter_width=2, 
                        residual_channels=32, 
                        dilation_channels=32, 
                        skip_channels=256, 
                        use_biases=True, 
                        use_batch_norm=True, 
                        dilations=[1, 2, 4, 8, 16, 32, 64, 128], 
                        random_seed=1234,
                        MIMO=False)
    '''
    The first two elements are your train and test set
    batch_size: how many sequences are passed to the GPU at once. The bigger batch_size is, the faster training will go. However, too big and you will run out of memory
    max_epochs: maximum number of epochs to train, 10-15 should do 
    plot: leave False
    train_fraction: what portion of the training set to use for training, the rest will be used for validation
    '''
    wavenet.train_and_predict(center_node_train, center_node_test, batch_size=128, max_epochs=1, plot=False, train_fraction=0.8)
     
