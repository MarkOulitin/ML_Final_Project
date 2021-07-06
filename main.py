import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import keras.utils
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from dataset_reader import read_data


def buildModel(input_dim, classes_count):
    inputs = tf.keras.Input(shape=(input_dim,))
    model = layers.Dense(32, activation="relu", name="layer1")(inputs)
    model = layers.Dense(32, activation="relu", name="layer2")(model)
    model = layers.Dense(32, activation="relu", name="layer3")(model)
    model = layers.Dense(32, activation="relu", name="layer4")(model)
    model = layers.Dense(classes_count, activation="relu", name="layer5")(model)
    model = keras.Model(inputs=inputs, outputs=model)
    return model, inputs


temp = True


def buildLoss(alpha, epsilon, x):
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_true, y_pred):
        global temp
        if temp:
            temp = False
            print(y_true)
            print(y_pred)
            print(x)
        return cross_entropy_loss(y_true, y_pred)

    return loss


def main():
    data, labels, classes_count, input_dim = read_data('waveform-noise.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    model, inputs = buildModel(input_dim, classes_count)
    loss = buildLoss(1, 1e-3, inputs)
    model.compile(
        loss=loss,
        optimizer='adam'
    )

    model.fit(X_train, y_train)
    # def custom_loss_wrapper(input_tensor):
    #     def custom_loss(y_true, y_pred):
    #         return K.binary_crossentropy(y_true, y_pred) + K.mean(input_tensor)
    #
    #     return custom_loss

    # input_tensor = layers.Input(shape=(10,))
    # hidden = layers.Dense(100, activation='relu')(input_tensor)
    # out = layers.Dense(1, activation='sigmoid')(hidden)
    # out.compile(loss=custom_loss_wrapper(input_tensor), optimizer='adam')
    #
    # X = np.random.rand(1000, 10)
    # y = np.random.randint(2, size=1000)
    # out.test_on_batch(X, y)  # => 1.1974642
    #
    # X *= 1000
    # out.test_on_batch(X, y)  # => 511.15466
    return


if __name__ == "__main__":
    main()
