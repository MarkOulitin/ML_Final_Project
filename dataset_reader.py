import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

dataset_dir = './classification_datasets/'


def showClasses(df, column_name):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    instances_by_class = df.groupby([column_name]).size()
    labels = instances_by_class.index.tolist()
    sizes = list(instances_by_class)
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def split_to_data_and_target(df: pd.DataFrame):
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    X = X.astype('float32')
    y = y.astype('int32')
    return X, y


def preprocessing(df):
    # print(list(df.columns.values), df.shape[0])
    X, y = split_to_data_and_target(df)
    classes_count = df[df.columns[-1]].nunique()
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1
    y = b
    return X, y, classes_count
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # return X_train, X_test, y_train, y_test


def read_data(filename):
    df = pd.read_csv(dataset_dir + filename)
    X, y, classes_count = preprocessing(df)
    # X_train, X_test, y_train, y_test = preprocessing(df)
    # print(f"{filename} => Finish preprocessing")
    # return X_train, X_test, y_train, y_test
    input_dim = X.shape[1]
    return X, y, classes_count, input_dim


if __name__ == '__main__':
    read_data('waveform-noise.csv')


def get_files():
    count = 0
    for filename in os.listdir(dataset_dir):
        if fetch_dataset_by_data(filename):
            count += 1
    print(f'total {count}')


def fetch_dataset_by_data(filename):
    df = pd.read_csv(dataset_dir + filename)
    types = df.dtypes[df.dtypes == 'float64']
    if len(types) == (len(df.dtypes) - 1) and len(df.index) > 1000:
        print(filename)
        return True
    return False