import os
from pprint import pprint

import numpy as np
import pandas as pd

dataset_dir = './classification_datasets/'


def convert_enum_to_number(y):
    """
    description: convert array of countable elements to array of representing integers
    example: ['cat', 'dog', 'cat', 'banana'] => [0, 1, 0, 2]
    :param y: numpy array
    :return: numpy array of integers representing each element as enum element
    """
    labels = np.unique(y)
    hash_map_labels = dict(zip(labels, np.arange(len(labels))))
    return np.array(list(map(lambda label: hash_map_labels[label], y)))


def split_to_data_and_target(df: pd.DataFrame):
    """
    :param df:
    :return: numpy ndarray of (instances X attributes) and numpy array of (instances X class_number)
    """
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    X = X.astype('float32')
    y = convert_enum_to_number(y)
    return X, y


def preprocessing(df):
    """
    :param df:
    :return: numpy ndarray of (instances X attributes) and numpy array of (instances X class_number) and amount of classes
    """
    X, y = split_to_data_and_target(df)
    classes_count = df[df.columns[-1]].nunique()
    return X, y, classes_count


def read_data(filename):
    """
    :param filename of dataset in ./classification_datasets/ directory
    :return: numpy ndarray of (instances X attributes) and numpy array of (instances X class_number)
             and amount of classes and amount of attributes
    """
    df = pd.read_csv(dataset_dir + filename)
    X, y, classes_count = preprocessing(df)
    input_dim = X.shape[1]
    return X, y, classes_count, input_dim


def get_files():
    """
    prints the datasets names in ./classification_datasets/ directory that matches the condition of fetch_dataset_by_data function
    :return:
    """
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
    """
    :param filename of dataset in ./classification_datasets/ directory
    :return: filename of input, amount of entries in dataset, and amount of classes if:
                the type of all columns of the input dataset is float64 except the last column and
                amount of entries in the dataset is more than 1,000
            otherwise returns None
    """
    df = pd.read_csv(dataset_dir + filename)
    types = df.dtypes[df.dtypes == 'float64']
    if len(types) == (len(df.dtypes) - 1) and len(df.index) > 1000:
        class_count = df[df.columns[-1]].nunique()
        return filename, len(df.index), class_count
    return None


if __name__ == '__main__':
    read_data('waveform-noise.csv')
    # get_files()
