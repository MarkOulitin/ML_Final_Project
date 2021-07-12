import os
from pprint import pprint

import numpy as np
import pandas as pd

dataset_dir = './classification_datasets/'


def convert_enum_to_number(y):
    labels = np.unique(y)
    hash_map_labels = dict(zip(labels, np.arange(len(labels))))
    return np.array(list(map(lambda label: hash_map_labels[label], y)))


def split_to_data_and_target(df: pd.DataFrame):
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    X = X.astype('float32')
    y = convert_enum_to_number(y)
    return X, y


def preprocessing(df):
    X, y = split_to_data_and_target(df)
    classes_count = df[df.columns[-1]].nunique()
    return X, y, classes_count


def read_data(filename):
    df = pd.read_csv(dataset_dir + filename)
    X, y, classes_count = preprocessing(df)
    input_dim = X.shape[1]
    return X, y, classes_count, input_dim


def get_files():
    datasets = []
    for filename in os.listdir(dataset_dir):
        dataset = fetch_dataset_by_data(filename)
        if dataset is not None:
            datasets.append(dataset)

    def compare_dataset_by_size(d):
        return d[1]

    datasets.sort(key=compare_dataset_by_size)
    datasets_top_20 = datasets[:20]
    dataset_names = list(map(lambda d: d[0], datasets_top_20))
    print(f'Total {len(dataset_names)}')
    pprint(dataset_names)


def fetch_dataset_by_data(filename):
    df = pd.read_csv(dataset_dir + filename)
    types = df.dtypes[df.dtypes == 'float64']
    if len(types) == (len(df.dtypes) - 1) and len(df.index) > 1000:
        class_count = df[df.columns[-1]].nunique()
        return filename, len(df.index), class_count
    return None


if __name__ == '__main__':
    read_data('waveform-noise.csv')
    # get_files()
