from WaveNet2 import WaveNet2, align_time_series
from MLP import MLP
import matplotlib.pyplot as plt
from ercot_data_interface import ercot_data_interface
import numpy as np
import cPickle as pickle
from sklearn.preprocessing import MinMaxScaler


TEST_NODES = ['CAL_CALGT1', 'KING_KINGNW', 'MAR_MARSFOG1', 'CALAVER_JKS1', 'RAYB_G78910', 'WOO_WOODWRD1', 'DECKER_DPG2', 'CEL_CELANEG1', 'FO_FORMOSG3', 'NUE_NUECESG7']
settings = ['auto_regressive', 'MIMO']

if __name__ == '__main__':
    forecast_horizons = [1, 24, 48, 72, 96, 120, 144, 168]
    modes = ['Auto-regressive', 'Conditional']
    with open('./wavenet_results.csv', 'w+') as f1:
        f1.write('Node\tType\tmae\tmase\thits\n')
        for m in modes:
            for f in forecast_horizons:
                for test_node in TEST_NODES:
                    print m, f, test_node
                    with open('/mnt/hdd1/ERCOT/' + test_node + '_train.pkl', 'r') as f2:
                        train = pickle.load(f2)
                    with open('/mnt/hdd1/ERCOT/' + test_node + '_test.pkl', 'r') as f2:
                        test = pickle.load(f2)

                    train_a = np.expand_dims(train[:, 0], 1)
                    test_a = np.expand_dims(test[:, 0], 1)
                    series = np.vstack((train_a, test_a))
                    mimo_series = np.vstack((train, test))
                    wavenet2 = WaveNet2(initial_filter_width=48, 
                                filter_width=2, 
                                dilation_channels=32, 
                                dilations=[1, 2, 4, 8, 16, 32, 64],
                                forecast_horizon=f,
                                random_seed=22943)
                    mlp = MLP(look_back=175, random_seed=1234, log_difference=False, forecast_horizon=f)
                    if m=='Auto-regressive':
                        predicted_w, actual_w = wavenet2.train(series, 5000)
                        predicted_m, actual_m = mlp.train(series, epochs=5000)
                    else:
                        predicted_w, actual_w = wavenet2.train(mimo_series, 5000)
                        predicted_m, actual_m = mlp.train(mimo_series, epochs=5000)
                    
                    
                    aligned = align_time_series([actual_w, predicted_w, predicted_m])
                    #RMSE
                    mse_wavenet =np.mean(np.square(aligned[0]-aligned[1]))
                    mse_mlp = np.mean(np.square(aligned[0]-aligned[2]))
                    mse_trivial = np.mean(np.square(aligned[0][:-f]-aligned[0][f:]))
                    #U statistic
                    mase_wavenet = np.sqrt(mse_wavenet/mse_trivial)
                    mase_mlp = np.sqrt(mse_mlp/mse_trivial)
                    print 'WaveNet', 'MLP', 'Trivial'
                    print mse_wavenet, mse_mlp, mse_trivial
                    print mase_wavenet, mase_mlp, '1'


                    f1.write('%s\t%s\t%s\%f\t%f\t%f\t%f\n' % (m, test_node, f, mse_wavenet, mse_mlp, mase_wavenet, mase_mlp))

