from VAR import VAR
import matplotlib.pyplot as plt
from ercot_data_interface import ercot_data_interface
import numpy as np
import cPickle as pickle


TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']

if __name__ == '__main__':
    ercot = ercot_data_interface()
    with open('./VAR_results.csv', 'w+') as f1:
        f1.write('Node\tmae\tmase\thits\n')
        for test_node in TEST_NODES:
            print test_node
            with open('/mnt/hdd1/ERCOT/' + test_node + '_train.pkl', 'r') as f2:
                train = pickle.load(f2)
            with open('/mnt/hdd1/ERCOT/' + test_node + '_test.pkl', 'r') as f2:
                test = pickle.load(f2)

            var = VAR()
            var.train(train, look_back=2)
            predicted, actual = var.predict(test)
            mae, mase, hits = var.print_statistics(predicted, actual)
            f1.write('%s\t%f\t%f\t%f\n' % (test_node, mae, mase, hits))

