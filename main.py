import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import keras.utils
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import tensorflow as tf
from Model_VatCustomFit import ModelVatCustomFit
from dataset_reader import read_data
from Datasets import datasets_names
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import uniform
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix

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
        model = ModelVatCustomFit(inputs=in_layer, outputs=layer5, method=method, epsilon=epsilon, alpha=alpha, xi=xi)
        return model

    return buildModel


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
        clf = RandomizedSearchCV(model, distributions, random_state=0, cv=CV_INNER_N_ITERATIONS)
        result = clf.fit(X_train, y_train)
        best_model = result.best_estimator_
        best_model.model.__class__ = ModelVatCustomFit
        train_time = best_model.model.train_time
        y_predict = best_model.predict(X_test)
        TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict, [])
        # may be predict proba?
        # not needed because we have last layer softmax aka returns array of probabilities - what roc_auc expects
        AUC_ROC = roc_auc_score(y_test, y_predict)


if __name__ == "__main__":
    main('blah')
