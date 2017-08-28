from WaveNet import WaveNet
from WaveNet2 import WaveNet2
import matplotlib.pyplot as plt
from ercot_data_interface import ercot_data_interface
import numpy as np
import cPickle as pickle


TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']
settings = ['auto_regressive', 'MIMO']

if __name__ == '__main__':
    np.random.seed(22943)
    random_seeds = np.random.randint(1, 10000, (10, 1))
    ercot = ercot_data_interface()
    with open('./wavenet_results.csv', 'w+') as f1:
        f1.write('Node\tType\tmae\tmase\thits\n')
        for test_node in TEST_NODES:
            print test_node
            with open('/mnt/hdd1/ERCOT/' + test_node + '_train.pkl', 'r') as f2:
                train = pickle.load(f2)
            with open('/mnt/hdd1/ERCOT/' + test_node + '_test.pkl', 'r') as f2:
                test = pickle.load(f2)


            maes = []
            mases = []
            hitss = []
            for r in random_seeds:
                train_a =train[:, 0]
                test_a = test[:, 0]
                series = np.vstack(train_a, test_a)
                wavenet2 = WaveNet2(initial_filter_width=48, 
                    filter_width=2, 
                    dilation_channels=32, 
                    use_batch_norm=False,
                    dilations=[1, 2, 4, 8, 16, 32, 64],
                    random_seed=r[0])
                
            mae, mase, hits = wavenet.train(series, 5000)
            maes.append(mae)
            mases.append(mase)
            hitss.append(hits)
            mae_avg = np.mean(maes)
            mae_sd = np.std(maes)
            mase_avg = np.mean(mases)
            mase_sd = np.std(mases)
            hits_avg = np.mean(hitss)
            hits_sd = np.std(hitss)
            f1.write('%s\t%f+/-%f\t%f+/-%f\t%f+/-%f\n' % (test_node, s, mae_avg, mae_sd, mase_avg, mase_sd, hits_avg, hits_sd))

