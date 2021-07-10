import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, precision_score, accuracy_score
from scipy.stats import uniform

from Model_VatCustomFit import ModelVatCustomFit
from dataset_reader import read_data
from Datasets import get_datasets_names
from utils import save_to_dict, create_dict, save_to_csv, setup, merge_results

NaN = float('nan')

CV_OUTER_N_ITERATIONS = 10
CV_INNER_N_ITERATIONS = 3
METRIC_AVERAGE = 'macro'


def safe_div(numerator, denominator, default):
    if np.isscalar(denominator) and denominator == 0:
        return default
    return numerator / denominator


# taken from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def compute_tpr_fpr_acc(y_true, y_pred, labels, average):
    if average != 'micro' and average != 'macro' and average != 'binary':
        raise ValueError(f'invalid average argument \'{average}\'')

    # print(f'labels: {labels}')
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.diag(conf_mat)
    all_sum = conf_mat.sum()

    # print(f'y_true: {np.transpose(y_true[:10])}')
    # print(f'y_pred: {np.transpose(y_pred[:10])}')
    # print(conf_mat)

    if average == 'binary':
        if conf_mat.shape != (2, 2):
            raise ValueError(f'binary average requested for non-binary confusion matrix ({conf_mat.shape})')

        FP = conf_mat[0, 1]
        TP = conf_mat[1, 1]
        FN = conf_mat[1, 0]
        TN = conf_mat[0, 0]
    else:
        FP = conf_mat.sum(axis=0) - diag
        FN = conf_mat.sum(axis=1) - diag
        TP = diag
        TN = all_sum - (FP + FN + TP)

    # print(f'FP {FP}, {FP.sum()}')
    # print(f'FN {FN}, {FN.sum()}')
    # print(f'TP {TP}, {TP.sum()}')
    # print(f'TN {TN}, {TN.sum()}')

    if average == 'micro':
        FP = FP.sum()
        FN = FN.sum()
        TP = TP.sum()
        TN = TN.sum()

    # True positive rate
    TPR = safe_div(TP, (TP + FN), NaN)
    # False positive rate
    FPR = safe_div(FP, (FP + TN), NaN)
    PRECISION = safe_div(TP, (TP + FP), NaN)

    if average == 'macro':
        TPR = np.average(TPR)
        FPR = np.average(FPR)
        PRECISION = np.average(PRECISION)

    if average == 'micro':
        ACC = TP / all_sum
    elif average == 'macro':
        ACC = TP.sum() / all_sum
    else:
        ACC = (TP + TN) / all_sum

    return TPR, FPR, ACC, PRECISION


def buildLayers(in_layer, dropout_rate=0.5, isDropout=False):
    layer1 = layers.Dense(32, activation="relu", name="layer1")(in_layer)
    if isDropout:
        layer1 = layers.Dropout(dropout_rate)(layer1)
    layer2 = layers.Dense(32, activation="relu", name="layer2")(layer1)
    if isDropout:
        layer2 = layers.Dropout(dropout_rate)(layer2)
    layer3 = layers.Dense(32, activation="relu", name="layer3")(layer2)
    if isDropout:
        layer3 = layers.Dropout(dropout_rate)(layer3)
    layer4 = layers.Dense(32, activation="relu", name="layer4")(layer3)
    if isDropout:
        layer4 = layers.Dropout(dropout_rate)(layer4)
    return layer4


def configHyperModelFactory(method, input_dim, classes_count):
    def buildModel(epsilon=1e-3, alpha=1, dropout_rate=0.2):
        xi = 1e-6
        if classes_count > 2:
            out_units_count = classes_count
            base_loss = losses.CategoricalCrossentropy()
            activation = 'softmax'
        else:
            out_units_count = 1
            base_loss = losses.BinaryCrossentropy()
            activation = 'sigmoid'
        optimizer = optimizers.Adam(learning_rate=1e-3)

        in_layer = layers.Input(shape=(input_dim,))
        layer4 = buildLayers(in_layer, dropout_rate, method == 'Dropout')
        layer5 = layers.Dense(out_units_count, activation=activation, name="layer5")(layer4)
        model = ModelVatCustomFit(
            inputs=in_layer,
            outputs=layer5,

            method=method,
            epsilon=epsilon,
            alpha=alpha,
            xi=xi
        )
        model.compile(loss=base_loss, optimizer=optimizer)
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
    # merge_results(results_filename)
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
    outer_cv = StratifiedKFold(n_splits=CV_OUTER_N_ITERATIONS)

    print(f'Working on: {dataset_name} with Algo: {method}')
    for iteration, (train_indexes, test_indexes) in enumerate(outer_cv.split(data)):
        X_train, X_test = data[train_indexes, :], data[test_indexes, :]
        y_train, y_test = labels[train_indexes], labels[test_indexes]
        model = KerasClassifier(build_fn=model_factory, epochs=10, batch_size=32, verbose=0)
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
        report = report_performance(data, y_predict, y_predict_proba, y_test, best_model, classes_count)
        if method == 'Dropout':
            hp_values = 'dropout_rate = ' + str(np.round(result.best_params_['dropout_rate'], 3))
        else:
            alpha_str = 'alpha = ' + str(np.round(result.best_params_['alpha'], 3))
            eps_str = 'epsilon = ' + str(np.round(result.best_params_['epsilon'], 3))
            hp_values = alpha_str + '\n' + eps_str
        save_to_dict(performance, iteration + 1, hp_values, *report)
        print(f'Dataset {dataset_name} -- Done {iteration + 1} iteration')
    save_to_csv(performance, dataset_name + "_" + method)


def some_test():
    setup()
    evaluate('waveform-noise.csv', 'Dropout')
    evaluate('titanic.csv', 'Dropout')


def report_performance(dataset, y_predict, y_predict_proba, y_test, best_model, classes_count, is_print=False):
    y_test_one_hot = keras.utils.to_categorical(y_test)
    if classes_count == 2:
        metrics_average = 'binary'
        y_positive_proba = y_predict_proba[:, 1]
    else:
        metrics_average = METRIC_AVERAGE

    TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict, labels=best_model.classes_, average=metrics_average)

    if classes_count == 2:
        # average cannot be binary here since it raises an exception
        if len(np.unique(y_test)) != 2:
            AUC_ROC = NaN
        else:
            AUC_ROC = roc_auc_score(y_test, y_positive_proba, average=METRIC_AVERAGE, multi_class='ovo')

        if not y_test.any():
            AUC_Precision_Recall = NaN
        else:
            AUC_Precision_Recall = average_precision_score(y_test, y_positive_proba, average=METRIC_AVERAGE)
    else:
        AUC_ROC = roc_auc_score(y_test, y_predict_proba, average=METRIC_AVERAGE, multi_class='ovo')

        # in our case, it will perceive it as multi-label.
        AUC_Precision_Recall = average_precision_score(y_test_one_hot, y_predict_proba, average=METRIC_AVERAGE)

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


def statistic_test(data_filename, amount_of_datasets, amount_of_algorithms):
    print('No yet implemented')


if __name__ == "__main__":
    device_name = "/cpu:0"
    with tf.device(device_name):
        main()
