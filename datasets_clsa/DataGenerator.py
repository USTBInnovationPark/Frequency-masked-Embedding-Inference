import os
import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from datasets_clsa.DatasetEnum import DatasetName


class DefaultGenerator(Dataset):
    def __init__(self,
                 data_name: DatasetName,
                 flag: str,
                 x_len: int = 0):
        super(DefaultGenerator, self).__init__()
        assert flag in ["train", "test", "val"]

        file_name = data_name.value
        raw_data = torch.load(os.path.join(file_name, "{}.pt".format(flag)))
        train_data = torch.load(os.path.join(file_name, "train.pt"))

        # only single variable be considered
        if x_len > 0:
            mean = train_data["samples"][:, :1, :x_len].mean(dim=[0, 1])
            std = train_data["samples"][:, :1, :x_len].std(dim=[0, 1])
            self.x_data = raw_data["samples"][:, :1, :x_len]
        else:
            mean = train_data["samples"][:, :1, :].mean(dim=[0, 1])
            std = train_data["samples"][:, :1, :].std(dim=[0, 1])
            self.x_data = raw_data["samples"][:, :1, :]
        self.x_data = (self.x_data - mean)/(std+1e-7)
        self.x_data = self.x_data.transpose(-1, -2)
        self.y_data = raw_data["labels"]

    def __getitem__(self, index):
        # x_data.shape = (length, 1)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]


def load_UCR(data_name: str):
    train_file = os.path.join('/home/fuen/DeepLearningProjects/TimesURL/src/datasets/UCR', data_name, data_name + "_TRAIN.tsv")
    test_file = os.path.join('/home/fuen/DeepLearningProjects/TimesURL/src/datasets/UCR', data_name, data_name + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float32)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float32)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if data_name in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std
    # extend dim to NTC
    train, test = train[..., np.newaxis], test[..., np.newaxis]
    train_missing = np.isnan(train).all(axis=-1).any(axis=0)
    test_missing = np.isnan(test).all(axis=-1).any(axis=0)
    if train_missing[0] or train_missing[-1]:
        train = _center_vary_length_series(train)
    if test_missing[0] or test_missing[-1]:
        test = _center_vary_length_series(test)
    train = train[~np.isnan(train).all(axis=2).all(axis=1)]
    test = test[~np.isnan(test).all(axis=2).all(axis=1)]
    train = _set_nan2zero(train)
    test = _set_nan2zero(test)
    train_dataset = TensorDataset(torch.from_numpy(train).to(torch.float), torch.from_numpy(train_labels).to(torch.long))
    test_dataset = TensorDataset(torch.from_numpy(test).to(torch.float), torch.from_numpy(test_labels).to(torch.long))
    cls_num = np.max(train_labels) + 1
    return train_dataset, test_dataset, cls_num


def _center_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def _set_nan2zero(x):
    x[np.isnan(x)] = 0
    return x


if __name__ == '__main__':
    files = os.listdir("/home/fuen/DeepLearningProjects/TimesURL/src/datasets/UCR")
    data = load_UCR("AllGestureWiimoteX")


