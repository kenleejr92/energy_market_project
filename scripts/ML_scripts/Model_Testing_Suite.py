__author__ = 'kenlee'

import os
from Keras_Testing import Keras_NN

DATES = [('2011-01-01', '2011-12-31'),
         ('2012-01-01', '2012-12-31'),
         ('2013-01-01', '2013-12-31'),
         ('2014-01-01', '2014-12-31'),
         ('2015-01-01', '2015-12-31')]
LOAD_ZONES = ['LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'LZ_HOUSTON']

os.chdir('../test_results')
f = open('Model_results.csv', 'w+')
f.write('Model,Zone,Year,MAPE,TheilU1,TheilU2\n')
kNN = Keras_NN()
for sd, ed in DATES:
    kNN.query_db(sd, ed)
    for lz in LOAD_ZONES:
        # Test MLP with Model A
        kNN.load_data(lz, 'A')
        kNN.create_model(hidden_layers=30, type='MLP')
        kNN.train_model(epochs=150)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('MLPA', lz, sd[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))


        # Test Recurrent Neural Networks
        kNN.load_data(lz, 'LSTM')
        kNN.create_model(hidden_layers=30, type='SimpleRNN')
        kNN.train_model(epochs=50)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('SimpleRNN', lz, sd[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))

        # Test LSTM with Model LSTM
        kNN.load_data(lz, 'LSTM')
        kNN.create_model(hidden_layers=30, type='LSTM')
        kNN.train_model(epochs=50)
        kNN.predict()
        kNN.compute_metrics()
        f.write('%s,%s,%s,%f,%f,%f\n' % ('LSTM', lz, sd[:4], kNN.MAPE, kNN.TheilU1, kNN.TheilU2))
f.close()