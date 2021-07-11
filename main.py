import datetime
import os
import time
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

from KerasClassifierOur import KerasClassifierOur
from Model_VatCustomFit import ModelVatCustomFit
from dataset_reader import read_data
from Datasets import get_datasets_names
from utils import save_to_dict, create_dict, save_to_csv, setup, merge_results

NaN = float('nan')

CV_OUTER_N_ITERATIONS = 10
CV_INNER_N_ITERATIONS = 3
EPOCHS = 10
N_RANDOM_SEARCH_ITERS = 50
METRIC_AVERAGE = 'macro'


def safe_div(numerator, denominator, default):
    if np.isscalar(denominator) and denominator == 0:
        return default
    return numerator / denominator


# code adopted from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
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


def start_evaluation(method, should_save):
    setup()
    datasets_names = get_datasets_names()
    # methods = ['Article', 'OUR', 'Dropout']
    amount_of_datasets = len(datasets_names)
    for iteration, dataset_name in enumerate(datasets_names):
        # for method in methods:
        evaluate(dataset_name, method, should_save)
        print(f'Done processing {iteration + 1} datasets from {amount_of_datasets}')
    results_filename = 'Results.xlsx'
    # merge_results(results_filename)
    # statistic_test(results_filename, len(datasets_names), len(methods))


def evaluate(
        dataset_name,
        method,
        should_save,
        n_cv_outer_splits=CV_OUTER_N_ITERATIONS,
        n_cv_inner_splits=CV_INNER_N_ITERATIONS,
        epochs=EPOCHS,
        n_random_search_iters=N_RANDOM_SEARCH_ITERS
):
    # performance = create_dict(dataset_name, method)
    data, labels, classes_count, input_dim = read_data(dataset_name)
    if method == 'Dropout':
        distributions = dict(dropout_rate=uniform(loc=1e-6, scale=1 - 1e-6))
    else:
        distributions = dict(alpha=np.linspace(0, 2, 101),
                             epsilon=uniform(loc=1e-6, scale=2e-3))
    model_factory = configHyperModelFactory(method, input_dim, classes_count)
    outer_cv = StratifiedKFold(n_splits=n_cv_outer_splits)

    print(f'Working on: {dataset_name} with Algo: {method}')
    for iteration, (train_indexes, test_indexes) in enumerate(outer_cv.split(data, labels)):
        performance = create_dict(dataset_name, method)
        X_train, X_test = data[train_indexes, :], data[test_indexes, :]
        y_train, y_test = labels[train_indexes], labels[test_indexes]
        model = KerasClassifierOur(
            num_classes=classes_count,
            build_fn=model_factory,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        clf = RandomizedSearchCV(
            model,
            param_distributions=distributions,
            n_iter=1,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=n_cv_inner_splits),
            random_state=0
        )

        print(
            f'Starting iteration {iteration + 1}/{n_cv_outer_splits} at {datetime.datetime.now():%H:%M:%S} '
            f'on dataset \'{dataset_name}\', algorithm variant \'{method}\'',
            end='', flush=True
        )
        fit_start_time = time.time()

        result = clf.fit(X_train, y_train)

        fit_time_delta = time.time() - fit_start_time
        print(
            f'\rFinished iteration {iteration + 1}/{n_cv_outer_splits} at {datetime.datetime.now():%H:%M:%S} '
            f'on dataset \'{dataset_name}\', algorithm variant \'{method}\', '
            f'time took: {format_timedelta(datetime.timedelta(seconds=fit_time_delta))}'
        )

        best_model = result.best_estimator_
        y_predict = best_model.predict(X_test)
        y_predict_proba = best_model.predict_proba(X_test)
        report = report_performance(data, y_predict, y_predict_proba, y_test, best_model, classes_count)

        if should_save:
            if method == 'Dropout':
                hp_values = 'dropout_rate = ' + str(np.round(result.best_params_['dropout_rate'], 3))
            else:
                alpha_str = 'alpha = ' + str(np.round(result.best_params_['alpha'], 3))
                eps_str = f'epsilon = {np.round(result.best_params_["epsilon"] * 1000, 3)}e-3'
                hp_values = alpha_str + ', ' + eps_str
            save_to_dict(performance, iteration + 1, hp_values, *report)
            save_to_csv(performance, dataset_name + "_" + method)
        # print(f'Dataset {dataset_name} -- Done {iteration + 1} iteration')
    # save_to_csv(performance, dataset_name + "_" + method)


def test_evaluate(dataset_name, method):
    evaluate(
        dataset_name,
        method,
        n_cv_outer_splits=2,
        n_cv_inner_splits=2,
        epochs=1,
        n_random_search_iters=1
    )


def some_test():
    setup()
    method = 'Dropout'
    test_evaluate('mfeat-karhunen.csv', method)
    test_evaluate('titanic.csv', method)


def report_performance(dataset, y_predict, y_predict_proba, y_test, best_model, classes_count, is_print=False):
    y_test_one_hot = keras.utils.to_categorical(y_test)
    if classes_count == 2:
        metrics_average = 'binary'
        y_positive_proba = y_predict_proba[:, 1]
    else:
        metrics_average = METRIC_AVERAGE

    TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict, labels=best_model.classes_,
                                                   average=metrics_average)

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


def setup_gpu(gpu_mem_limit):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit - 1024)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('Physical GPUs:', gpus)
            print('Logical GPUs: ', logical_gpus)
            return tf.device('/device:GPU:0')
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def parse_args():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('algovar')
    args_parser.add_argument('-cpu', default=False, required=False, const=True, action='store_const')
    args_parser.add_argument(
        '-no-save', default=False, required=False,
        const=True, action='store_const', dest='no_save'
    )
    args_parser.add_argument('--gpu-mem-limit', type=int, default=None, required=False)
    args = args_parser.parse_args()
    return args


def choose_device(args):
    device = None
    if args.cpu:
        device = tf.device('/CPU:0')
    elif args.gpu_mem_limit is not None:
        device = setup_gpu(args.gpu_mem_limit)
    return device


def run_on_device(method, device, should_save):
    if device is None:
        start_evaluation(method, should_save)
    else:
        with device:
            start_evaluation(method, should_save)


def main():
    args = parse_args()
    print(f'args:', args)
    if len(sys.argv) > 1:
        device = choose_device(args)
        run_on_device(args.algovar, device, not args.no_save)
    else:
        print('Add argument => 1 = Article, 2 = OUR, 3 = Dropout')


def format_timedelta(td):
    s = td.total_seconds()
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)
    if hours > 0:
        return f'{hours:02}:{minutes:02}:{seconds:05.2f}'
    elif minutes > 0:
        return f'{minutes:02}:{seconds:05.2f}'
    else:
        return f'{seconds:.2f}s'


if __name__ == "__main__":
    main()
