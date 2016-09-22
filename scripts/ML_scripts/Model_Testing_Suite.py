__author__ = 'kenlee'

import os
from Keras_Testing import Keras_NN

DATES = ['2013-12-31',
         '2014-12-31',
         '2015-12-31']
LOAD_ZONES = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']

os.chdir('../test_results')
f = open('Model_results.csv', 'w+')
f.write('Model,Zone,Year,MAPE,TheilU1,TheilU2\n')

kNN = Keras_NN(type='LSTM')
for ed in DATES:
    for lz in LOAD_ZONES:
        print(lz + ' ' + ed + ' ' + 'SimpleRNN')
        # Test Recurrent Neural Networks
        kNN.load_data(ed, lz)
        kNN.create_model(hidden_layers=30, type='SimpleRNN')
        kNN.train_model(epochs=50)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('SimpleRNN', lz, ed[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

        print(lz + ' ' + ed + ' ' + 'LSTM')
        kNN.create_model(hidden_layers=30, type='LSTM')
        kNN.train_model(epochs=50)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('LSTM', lz, ed[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

        print(lz + ' ' + ed + ' ' + 'StackedLSTM')
        kNN.create_model(hidden_layers=30, type='StackedLSTM')
        kNN.train_model(epochs=50)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('StackedLSTM', lz, ed[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

kNN = Keras_NN(type='MLP')
start_date = '2012-07-01'
for ed in DATES:
    kNN.query_db(start_date, ed)
    for lz in LOAD_ZONES:
        # Test MLP with Model A
        print(lz + ' ' + ed + ' ' + 'MLP')
        kNN.load_data(ed, lz)
        kNN.create_model(hidden_layers=30, type='MLP')
        kNN.train_model(epochs=100)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('MLPA', lz, ed[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))
f.close()