__author__ = 'kenlee'

import numpy as np

class Sequence_Scaler(object):

    def __init__(self):
        self.mean_train_x = None
        self.std_train_x = None
        self.mean_train_y = None
        self.std_train_y = None

    def scale_training_data(self, train_x, train_y):
        dim1, dim2, dim3 = train_x.shape
        train_x_combined = np.concatenate(train_x, axis=0)
        self.mean_train_x = np.mean(train_x_combined[:,0])
        self.std_train_x = np.std(train_x_combined[:,0])
        self.mean_train_y = np.mean(train_y)
        self.std_train_y = np.std(train_y)
        train_x_combined[:, 0] = (train_x_combined[:, 0] - self.mean_train_x)/self.std_train_x
        train_y = (train_y - self.mean_train_y)/self.std_train_y
        train_x = np.reshape(train_x_combined, (dim1, dim2, dim3))
        return train_x, train_y


    def scale_testing_data(self, test_x, test_y):
        dim1, dim2, dim3 = test_x.shape
        test_x_combined = np.concatenate(test_x, axis=0)
        test_x_combined[:, 0] = (test_x_combined[:, 0] - self.mean_train_x)/self.std_train_x
        test_x = np.reshape(test_x_combined, (dim1, dim2, dim3))
        test_y = (test_y - self.mean_train_y)/self.std_train_y
        return test_x, test_y

    def inverse_scale(self, y):
        return y*self.std_train_y + self.mean_train_y