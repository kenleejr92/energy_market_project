from WaveNet import WaveNet
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


            for s in settings:
                maes = []
                mases = []
                hitss = []
                for r in random_seeds:
                    train_a = np.expand_dims(train[:, 0], 1)
                    test_a = np.expand_dims(test[:, 0], 1)
                    if s == 'MIMO':
                        MIMO = True
                    else: 
                        MIMO = False
                    wavenet = WaveNet(forecast_horizon=1, 
                                        log_difference=True, 
                                        initial_filter_width=48, 
                                        filter_width=2, 
                                        residual_channels=64, 
                                        dilation_channels=64, 
                                        skip_channels=64, 
                                        use_biases=True, 
                                        use_batch_norm=False, 
                                        dilations=[1, 2, 4, 8, 16, 32], 
                                        random_seed=r,
                                        MIMO=MIMO)
                    if s == 'auto_regressive':
                        mae, mase, hits = wavenet.train_and_predict(train_a, test_a, batch_size=128, max_epochs=10, plot=False, train_fraction=0.8)
                    else:
                        mae, mase, hits = wavenet.train_and_predict(train, test, batch_size=128, max_epochs=10, plot=False, train_fraction=0.8)
                    maes.append(mae)
                    mases.append(mase)
                    hitss.append(hits)
                mae_avg = np.mean(maes)
                mae_sd = np.std(maes)
                mase_avg = np.mean(mases)
                mase_sd = np.std(mases)
                hits_avg = np.mean(hitss)
                hits_sd = np.std(hitss)
                f1.write('%s\t%s\t%f+/-%f\t%f+/-%f\t%f+/-%f\n' % (test_node, s, mae_avg, mae_sd, mase_avg, mase_sd, hits_avg, hits_sd))

