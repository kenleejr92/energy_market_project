from ercot_data_interface import ercot_data_interface
import os
import numpy as np
import cPickle as pickle

TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']
SAVE_DIR = '/mnt/hdd1/ERCOT'

if __name__ == '__main__':
    ercot = ercot_data_interface()
    for test_node in TEST_NODES:
        print test_node
        nodes = ercot.get_nearest_CRR_neighbors(test_node)
        train, test = ercot.get_train_test(nodes, normalize=False, include_seasonal_vectors=False)
        f1 = open(SAVE_DIR + '/' + test_node + '_train.pkl', 'w+')
        pickle.dump(train, f1)
        f2 = open(SAVE_DIR + '/' + test_node + '_test.pkl', 'w+')
        pickle.dump(test, f2)
        f1.close()
        f2.close()