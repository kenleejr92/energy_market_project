__author__ = 'kenlee'

import os
from Keras_Testing import Keras_NN

DATES = ['2012-12-31',
         '2013-12-31',
         '2014-12-31',
         '2015-12-31']
LOAD_ZONES = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']

os.chdir('../test_results')
f = open('Model_results.csv', 'w+')
f.write('Model,Zone,Year,MAPE,TheilU1,TheilU2\n')
start_date = '2011-01-01'
end_date = '2015-12-31'
kNN = Keras_NN(type='LSTM')
kNN.query_db(start_date, end_date)
for lz in LOAD_ZONES:
    # Test Recurrent Neural Networks
    kNN.load_data(lz)
    kNN.create_model(hidden_layers=30, type='SimpleRNN')
    kNN.train_model(epochs=10)
    kNN.predict()
    kNN.compute_metrics()
    f.write('%s,%s,%s,%f,%f,%f\n' % ('SimpleRNN', lz, start_date[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

    # Test LSTM with Model LSTM
    # kNN.load_data(lz, 'LSTM')
    kNN.create_model(hidden_layers=30, type='LSTM')
    kNN.train_model(epochs=10)
    kNN.predict()
    kNN.compute_metrics()
    f.write('%s,%s,%s,%f,%f,%f\n' % ('LSTM', lz, start_date[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

    # Test LSTM with Model LSTM
    # kNN.load_data(lz, 'LSTM')
    kNN.create_model(hidden_layers=30, type='StackedLSTM')
    kNN.train_model(epochs=20)
    kNN.predict()
    kNN.compute_metrics()
    f.write('%s,%s,%s,%f,%f,%f\n' % ('StackedLSTM', lz, start_date[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

kNN = Keras_NN(type='MLP')
kNN.query_db(start_date, end_date)
for lz in LOAD_ZONES:
    # Test MLP with Model A
    kNN.load_data(lz)
    kNN.create_model(hidden_layers=30, type='MLP')
    kNN.train_model(epochs=10)
    kNN.predict()
    kNN.compute_metrics()
    f.write('%s,%s,%s,%f,%f,%f\n' % ('MLPA', lz, start_date[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))
f.close()