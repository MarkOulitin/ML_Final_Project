import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix, average_precision_score
from scipy.stats import uniform

from Model_VatCustomFit import ModelVatCustomFit
from dataset_reader import read_data
from Datasets import datasets_names

CV_OUTER_N_ITERATIONS = 10
CV_INNER_N_ITERATIONS = 3


# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def compute_tpr_fpr_acc(y_true, y_pred, y_pred_probabilities):
    conf_mat = confusion_matrix(y_true, y_pred)
    FP = confusion_matrix.sum(axis=0) - np.diag(conf_mat)
    FN = confusion_matrix.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = confusion_matrix.values.sum() - (FP + FN + TP)

    # True positive rate
    TPR = TP / (TP + FN)
    # False positive rate
    FPR = FP / (FP + TN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    PRECISION = TP / (TP + FP)

    # pos_probs =y_pred_probabilities[:, ]
    return TPR, FPR, ACC, PRECISION


def configHyperModelFactory(method, input_dim, classes_count):
    def buildModel(epsilon=1e-3, alpha=1):
        xi = 1e-6
        in_layer = layers.Input(shape=(input_dim,))
        layer1 = layers.Dense(32, activation="relu", name="layer1")(in_layer)
        layer2 = layers.Dense(32, activation="relu", name="layer2")(layer1)
        layer3 = layers.Dense(32, activation="relu", name="layer3")(layer2)
        layer4 = layers.Dense(32, activation="relu", name="layer4")(layer3)
        layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")(layer4)
        model = ModelVatCustomFit(
            inputs=in_layer,
            outputs=layer5,

            method=method,
            epsilon=epsilon,
            alpha=alpha,
            xi=xi
        )
        model.compile(loss=losses.CategoricalCrossentropy(), optimizer=optimizers.Adam(learning_rate=1e-3))
        return model

    return buildModel


def calculate_inference_time(X, model):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    selected_indexes = indexes[:1000]
    x_test = X[selected_indexes, :]
    start_time = time.time()
    model.predict(x_test)
    return time.time() - start_time


def main(method):
    data, labels, classes_count, input_dim = read_data('waveform-noise.csv')
    distributions = dict(alpha=np.linspace(0, 2, 101),
                         epsilon=uniform(loc=1e-6, scale=2e-3))
    model_factory = configHyperModelFactory(method, input_dim, classes_count)
    outer_cv = KFold(n_splits=CV_OUTER_N_ITERATIONS)

    for train_indexes, test_indexes in outer_cv.split(data):
        X_train, X_test = data[train_indexes, :], data[test_indexes, :]
        y_train, y_test = labels[train_indexes], labels[test_indexes]
        model = KerasClassifier(build_fn=model_factory, epochs=5, batch_size=32, verbose=0)
        clf = RandomizedSearchCV(
            model,
            param_distributions=distributions,
            n_iter=50,
            scoring='accuracy',
            cv=CV_INNER_N_ITERATIONS,
            random_state=0
        )
        result = clf.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_predict = best_model.predict(X_test)
        y_predict_proba = best_model.predict_proba(X_test)
        print(y_predict.shape, y_predict[0])
        print(y_predict_proba.shape, y_predict_proba[0])
        # TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict, [])

        # may be predict proba?
        # not needed because we have last layer softmax aka returns array of probabilities - what roc_auc expects
        # AUC_ROC = roc_auc_score(y_test, y_predict)
        # AUC_Precision_Recall = average_precision_score(y_test, y_predict)
        # best_model.model.__class__ = ModelVatCustomFit
        # train_time = best_model.model.train_time
        # inference_time = calculate_inference_time(data, best_model)


def some_test(method):
    data, labels, classes_count, input_dim = read_data('waveform-noise.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    distributions = dict(alpha=np.linspace(0, 2, 101),
                         epsilon=uniform(loc=1e-6, scale=2e-3))
    model_factory = configHyperModelFactory(method, input_dim, classes_count)
    model = KerasClassifier(build_fn=model_factory, epochs=1, batch_size=32, verbose=0)
    # model.fit(X_train, y_train)
    clf = RandomizedSearchCV(
        model,
        param_distributions=distributions,
        n_iter=1,
        scoring='accuracy',
        cv=CV_INNER_N_ITERATIONS,
        random_state=0
    )
    result = clf.fit(X_train, y_train)
    print(result.refit_time_, result.best_estimator_.model.train_time)
    # best_model = result.best_estimator_
    # y_predict = best_model.predict(X_test)
    # y_predict_proba = best_model.predict_proba(X_test)
    # print(y_predict.shape, y_predict[0])
    # print(y_predict_proba.shape, y_predict_proba[0])


if __name__ == "__main__":
    some_test('blah')
