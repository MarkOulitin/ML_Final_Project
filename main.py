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

from BaseModel import BaseModel
from FCModel import CustomModel, kl_divergence
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


def build_vat_loss(model, inputs, epsilon, alpha, xi):
    cce = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_true, y_pred):
        r_vadvs = model.compute_rvadvs(inputs, y_true, epsilon, xi)
        y_hat_vadvs = model(inputs + r_vadvs)
        R_vadv = kl_divergence(y_true, y_hat_vadvs)
        return cce(y_true, y_pred) + alpha * R_vadv

    return loss


def main(method):
    data, labels, classes_count, input_dim = read_data('waveform-noise.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    epsilon = 1e-3
    alpha = 1
    xi = 1e-6
    # method, input_dim, classes_count, epsilon, alpha, xi
    # model = BaseModel(input_dim, classes_count)
    model = CustomModel(method, input_dim, classes_count, epsilon, alpha, xi)
    model.compile(
        loss=build_vat_loss(model, model.inputs, epsilon, alpha, xi),
        optimizer='adam',
        metrics=['accuracy', 'categorical_crossentropy']
    )
    model.fit(X_train, y_train, epochs=1)

    model.evaluate(X_test, y_test)
    return


if __name__ == "__main__":
    main('blah')
