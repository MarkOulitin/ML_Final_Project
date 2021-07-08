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

from Model_VatCustomFit import ModelVatCustomFit
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


# def build_vat_loss(model, inputs, epsilon, alpha, xi):
#     cce = tf.keras.losses.CategoricalCrossentropy()
#
#     def loss(y_true, y_pred):
#         r_vadvs = model.compute_rvadvs(inputs, y_true, epsilon, xi)
#         y_hat_vadvs = model(inputs + r_vadvs)
#         R_vadv = kl_divergence(y_true, y_hat_vadvs)
#         return cce(y_true, y_pred) + alpha * R_vadv
#
#     return loss


def main(method):
    data, labels, classes_count, input_dim = read_data('waveform-noise.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    epsilon = 1e-3
    alpha = 1
    xi = 1e-6
    # method, input_dim, classes_count, epsilon, alpha, xi
    # model = BaseModel(input_dim, classes_count)
    in_layer = layers.Input(shape=(input_dim,))
    layer1 = layers.Dense(32, activation="relu", name="layer1")(in_layer)
    layer2 = layers.Dense(32, activation="relu", name="layer2")(layer1)
    layer3 = layers.Dense(32, activation="relu", name="layer3")(layer2)
    layer4 = layers.Dense(32, activation="relu", name="layer4")(layer3)
    layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")(layer4)
    model = ModelVatCustomFit(inputs=in_layer, outputs=layer5, method=method, epsilon=epsilon, alpha= alpha, xi= xi)
    model.fit(X_train, y_train, epochs=1)

    # model.evaluate(X_test, y_test)
    return


if __name__ == "__main__":
    main('blah')
