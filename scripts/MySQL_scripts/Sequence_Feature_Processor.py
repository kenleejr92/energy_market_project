__author__ = 'kenlee'

from DAM_prices_by_SP import Feature_Processor, sample_month, work_day_or_holiday, encode_onehot, MONTHS_PER_YEAR, DAYS_PER_MONTH, HRS_PER_DAY
import numpy as np
import pandas as pd
from Sequence_Scaler import Sequence_Scaler

class Sequence_Feature_Processor(Feature_Processor):

    def __init__(self):
        self.sequence_scaler = None
        self.sequences = []
        self.targets = []
        self.train_features = []
        self.test_features = []
        self.val_features = []
        self.train_targets = []
        self.test_targets = []
        self.val_targets = []
        super(Sequence_Feature_Processor, self).__init__()

    def construct_feature_vector_matrix(self, lzhub, model_type):
        self.targets = []
        self.sequences =[]
        self.lzhub = lzhub + '_SPP'
        dflzhub = self.df[lzhub + '_SPP']
        features = []
        for dt, price in dflzhub.iteritems():
            features.append([price, work_day_or_holiday(dt), dt.hour, dt.weekday(), dt.month])
        feature_labels = ['Price', 'Holiday', 'Hour', 'Day', 'Month']
        self.features_df = pd.DataFrame(data=features, index=dflzhub.index, columns=feature_labels)
        self.features_df = encode_onehot(self.features_df, 'Day')
        self.features_df = encode_onehot(self.features_df, 'Month')
        self.features_df = encode_onehot(self.features_df, 'Hour')
        for dt, row in self.features_df.iterrows():
            pred_hour_index = self.features_df.index.get_loc(dt)
            if pred_hour_index - 7*24 >= 0:
                self.targets.append(self.features_df.iloc[pred_hour_index]['Price'])
                sequence = []
                sequence.append(self.features_df.iloc[pred_hour_index - 24])
                sequence.append(self.features_df.iloc[pred_hour_index - 25])
                sequence.append(self.features_df.iloc[pred_hour_index - 72])
                sequence.append(self.features_df.iloc[pred_hour_index - 96])
                # sequence = [self.features_df.iloc[pred_hour_index - 24*i] for i in np.arange(1, 8)]
                self.sequences.append(sequence)
        self.targets = np.array(self.targets)
        self.sequences = np.array(self.sequences)

    def train_test_validate(self, scaling='standard', train_size=0.6, test_size=0.2):
        self.train_features = []
        self.test_features = []
        self.val_features = []
        self.train_targets = []
        self.test_targets = []
        self.val_targets = []
        train_indices = []
        test_indices = []
        val_indices = []
        sequences = self.sequences
        targets = self.targets

        for i in range(MONTHS_PER_YEAR):
            train_i, test_i, val_i = sample_month(i, train_size, test_size, sequences.shape[0])
            train_indices = train_indices + train_i
            test_indices = test_indices + test_i
            val_indices = val_indices + val_i
        # train_indices = [i*HRS_PER_DAY for i in train_indices]
        # test_indices = [i*HRS_PER_DAY for i in test_indices]
        # val_indices = [i*HRS_PER_DAY for i in val_indices]
        for i in train_indices:
            self.train_features.append(sequences[i, :, :])
            self.train_targets.append(targets[i])
        self.train_features = np.array(self.train_features)
        self.train_targets = np.array(self.train_targets)
        # self.train_features = np.concatenate(self.train_features, axis=0)
        # self.train_targets = np.concatenate(self.train_targets, axis=0)
        for i in test_indices:
            self.test_features.append(sequences[i, :, :])
            self.test_targets.append(targets[i])
        # self.test_features = np.concatenate(self.test_features, axis=0)
        # self.test_targets = np.concatenate(self.test_targets, axis=0)
        self.test_features = np.array(self.test_features)
        self.test_targets = np.array(self.test_targets)
        for i in val_indices:
            self.val_features.append(sequences[i, :, :])
            self.val_targets.append(targets[i])
        # self.val_features = np.concatenate(self.val_features, axis=0)
        # self.val_targets = np.concatenate(self.val_targets, axis=0)
        self.val_features = np.array(self.val_features)
        self.val_targets = np.array(self.val_targets)

        self.sequence_scaler = Sequence_Scaler()
        self.train_features, self.train_targets = self.sequence_scaler.scale_training_data(self.train_features, self.train_targets)
        self.test_features, self.test_targets = self.sequence_scaler.scale_testing_data(self.test_features, self.test_targets)
        self.val_features, self.val_targets = self.sequence_scaler.scale_testing_data(self.val_features, self.val_targets)
        return self.train_features, self.train_targets, self.test_features, self.test_targets, self.val_features, self.val_targets

    def inverse_scale(self, y):
        dim1 = y.shape[0]
        y = np.reshape(y, (dim1,))
        return self.sequence_scaler.inverse_scale(y)



if __name__ == '__main__':
    sfp  = Sequence_Feature_Processor()
    sfp.query('2015-01-01', '2015-12-31')
    sfp.construct_feature_vector_matrix('LZ_WEST', 'LSTM')
    sfp.train_test_validate()
    # print(sfp.sequences)
    # print(sfp.targets)
    print(sfp.train_features)
    print(sfp.train_targets)