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
from Datasets import get_datasets_names
from utils import save_to_dict, create_dict, save_to_csv, setup, merge_results

CV_OUTER_N_ITERATIONS = 10
CV_INNER_N_ITERATIONS = 3


# taken from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def compute_tpr_fpr_acc(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)

    FP = FP.sum()
    FN = FN.sum()
    TP = TP.sum()
    TN = TN.sum()
    # True positive rate
    TPR = TP / (TP + FN)
    # False positive rate
    FPR = FP / (FP + TN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    PRECISION = TP / (TP + FP)

    return TPR, FPR, ACC, PRECISION


def configHyperModelFactory(method, input_dim, classes_count):
    def build_Dropout_Model(dropout_rate=0.2):
        in_layer = layers.Input(shape=(input_dim,))
        layer1 = layers.Dense(32, activation="relu", name="layer1")(in_layer)
        layer1 = layers.Dropout(dropout_rate)(layer1)
        layer2 = layers.Dense(32, activation="relu", name="layer2")(layer1)
        layer2 = layers.Dropout(dropout_rate)(layer2)
        layer3 = layers.Dense(32, activation="relu", name="layer3")(layer2)
        layer3 = layers.Dropout(dropout_rate)(layer3)
        layer4 = layers.Dense(32, activation="relu", name="layer4")(layer3)
        layer4 = layers.Dropout(dropout_rate)(layer4)
        layer5 = layers.Dense(classes_count, activation="softmax", name="layer5")(layer4)
        model = ModelVatCustomFit(
            inputs=in_layer,
            outputs=layer5,

            method='Dropout',
            epsilon=None,
            alpha=None,
            xi=None
        )
        model.compile(loss=losses.CategoricalCrossentropy(), optimizer=optimizers.Adam(learning_rate=1e-3))
        return model

    def build_VAT_Model(epsilon=1e-3, alpha=1):
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

    if method == 'Dropout':
        return build_Dropout_Model
    else:
        return build_VAT_Model


def calculate_inference_time(X, model):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    selected_indexes = indexes[:1000]
    x_test = X[selected_indexes, :]
    start_time = time.time()
    model.predict(x_test)
    return time.time() - start_time


def main():
    setup()
    datasets_names = get_datasets_names()
    methods = ['Article', 'OUR', 'Dropout']
    amount_of_datasets = len(datasets_names)
    for iteration, dataset_name in enumerate(datasets_names):
        for method in methods:
            evaluate(dataset_name, method)
        print(f'Done processing {iteration + 1} datasets from {amount_of_datasets}')
    results_filename = 'Results.xlsx'
    merge_results(results_filename)
    statistic_test(results_filename)


def evaluate(dataset_name, method):
    performance = create_dict(dataset_name, method)
    data, labels, classes_count, input_dim = read_data(dataset_name)
    if method == 'Dropout':
        distributions = dict(dropout_rate=uniform(loc=1e-6, scale=1-1e-6))
    else:
        distributions = dict(alpha=np.linspace(0, 2, 101),
                             epsilon=uniform(loc=1e-6, scale=2e-3))
    model_factory = configHyperModelFactory(method, input_dim, classes_count)
    outer_cv = KFold(n_splits=CV_OUTER_N_ITERATIONS)

    print(f'Working on: {dataset_name} with Algo: {method}')
    for iteration, (train_indexes, test_indexes) in enumerate(outer_cv.split(data)):
        X_train, X_test = data[train_indexes, :], data[test_indexes, :]
        y_train, y_test = labels[train_indexes], labels[test_indexes]
        model = KerasClassifier(build_fn=model_factory, epochs=1, batch_size=32, verbose=0)
        clf = RandomizedSearchCV(
            model,
            param_distributions=distributions,
            n_iter=1,
            scoring='accuracy',
            cv=CV_INNER_N_ITERATIONS,
            random_state=0
        )
        result = clf.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_predict = best_model.predict(X_test)
        y_predict_proba = best_model.predict_proba(X_test)
        report = report_performance(data, y_predict, y_predict_proba, y_test, best_model)
        if method == 'Dropout':
            hp_values = 'dropout_rate = ' + str(np.round(result.best_params_['dropout_rate'], 3))
        else:
            alpha_str = 'alpha = ' + str(np.round(result.best_params_['alpha'], 3))
            eps_str = 'epsilon = ' + str(np.round(result.best_params_['epsilon'], 3))
            hp_values = alpha_str + '\n' + eps_str
        save_to_dict(performance, iteration + 1, hp_values, *report)
        print(f'Done {iteration + 1} iteration')
    save_to_csv(performance, dataset_name + "_" + method)


def some_test():
    setup()
    evaluate('waveform-noise.csv', 'Dropout')


def report_performance(dataset, y_predict, y_predict_proba, y_test, best_model, is_print=False):
    TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict)
    y_test_one_hot = keras.utils.to_categorical(y_test)
    AUC_ROC = roc_auc_score(y_test_one_hot, y_predict_proba, average='micro')
    AUC_Precision_Recall = average_precision_score(y_test_one_hot, y_predict_proba, average='micro')
    train_time = best_model.model.train_time
    inference_time = calculate_inference_time(dataset, best_model)
    if is_print:
        print(f'Accuracy {ACC}',
              f'TPR {TPR}',
              f'FPR {FPR}',
              f'PRECISION {PRECISION}',
              f'AUC ROC {AUC_ROC}',
              f'AUC PRECISION RECALL {AUC_Precision_Recall}',
              f'TRAIN TIME {train_time}',
              f'INFERENCE TIME FOR 1000 INSTANCES {inference_time}', sep="\n")
    return TPR, FPR, ACC, PRECISION, AUC_ROC, AUC_Precision_Recall, train_time, inference_time


def statistic_test(data_filename):
    print('No yet implemented')


if __name__ == "__main__":
    some_test()
    # main()
