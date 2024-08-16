from enum import Enum
import os

root = os.path.dirname(__file__)


class DatasetName(Enum):
    ECG = root + "/ECG/"
    EMG = root + "/EMG/"
    EPI = root + "/Epilepsy/"
    FDA = root + "/FD-A/"
    FDB = root + "/FD-B/"
    GES = root + "/Gesture/"
    HAR = root + "/HAR/"
    SLE = root + "/SleepEEG/"


def get_cls_num(dataset):
    if dataset == "ECG":
        return 4
    elif dataset == "EMG":
        return 3
    elif dataset == "EPI":
        return 2
    elif dataset == "FDA":
        return 3
    elif dataset == "FDB":
        return 3
    elif dataset == "GES":
        return 8
    elif dataset == "HAR":
        return 6
    elif dataset == "SLE":
        return 5
    else:
        raise ValueError("Unknown dataset:{}".format(dataset))

