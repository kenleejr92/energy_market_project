__author__ = 'kenlee'
import cPickle as pickle
import Sequence_Scaler
import numpy as np

ed = '2014-12-31'
lz = 'LZ_NORTH'
f1 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_train_features.pkl' % (ed, lz), 'rb')
f2 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_test_features.pkl' % (ed, lz), 'rb')
f3 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_val_features.pkl' % (ed, lz), 'rb')
f4 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_train_targets.pkl' % (ed, lz), 'rb')
f5 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_test_targets.pkl' % (ed, lz), 'rb')
f6 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_val_targets.pkl' % (ed, lz), 'rb')
f7 = open('/home/kenlee/energy_market_project/scripts/normalized_datasets/%s_%s_seq_scaler.pkl' % (ed, lz), 'rb')

seq_train_features = pickle.load(f1)
seq_test_features = pickle.load(f2)
seq_val_features = pickle.load(f3)
seq_train_targets = pickle.load(f4)
seq_test_targets = pickle.load(f5)
seq_val_targets = pickle.load(f6)
seq_scaler = pickle.load(f7)

y_actual = seq_test_targets
dim1 = y_actual.shape[0]
y = np.reshape(y_actual, (dim1,))
print(seq_scaler.inverse_scale(y))