import datetime
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
from scipy.stats import uniform

from KerasClassifierOur import KerasClassifierOur
from VatKerasModel import VatKerasModel
from dataset_reader import read_data
from Datasets import get_datasets_names
from utils import save_to_dict, create_dict, save_to_csv, setup

NaN = float('nan')

CV_OUTER_N_ITERATIONS = 10
CV_INNER_N_ITERATIONS = 3
EPOCHS = 10
N_RANDOM_SEARCH_ITERS = 50
METRIC_AVERAGE = 'macro'


def safe_div(numerator, denominator, default):
    """
    Computes division between the 2 operands, but tries to avoid
    division by 0.
    If the denominator is 0, default is returned.
    Note: No handling for ndarrays or a compound data type
    :param numerator: The numerator
    :param denominator: The denominator
    :param default: The value returned if the denominator is 0
    :return: The division result or default if the denominator is 0
    """
    if np.isscalar(denominator) and denominator == 0:
        return default
    return numerator / denominator


# code adopted from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def compute_tpr_fpr_acc(y_true, y_pred, labels, average):
    """
    Computes TPR (true positive rate), FPR (false positive rate),
    accuracy and precision stats for the results (actual vs predicted)
    using the specified average kind.
    Accuracy is not affected by the average, it only has 1 kind of calculation.

    Notes:
    * see the following links for better understanding:
      - https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
      - https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    * Terminology:
      - TP: true positive
      - TN: true negative
      - FP: false positive
      - FN: false negative

    :param y_true: The actual label of the instances
    :param y_pred: The predicted label of the instances
    :param labels: The set of possible labels.
    It should be specified to make sure the confusion matrix is in the
    right size as some classes might not be present in test labels.
    :param average: The kind of average used to compute the stats.
    Can be one of the following:
      - 'micro': Sums up the global TP, FP, TN, FN and then calculate
      the stats.
      - 'macro': Calculate the stats for each class independently and
      return the average of the each stat.
      - 'binary': Treats TP, TN, FP and FN differently, each to it's own.
      It can get mixed when there are more than 2 classes as shown in the
      first link above.
    :return: Various stats as stated above
    """
    if average != 'micro' and average != 'macro' and average != 'binary':
        raise ValueError(f'invalid average argument \'{average}\'')

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.diag(conf_mat)
    all_sum = conf_mat.sum()

    if average == 'binary':
        if conf_mat.shape != (2, 2):
            raise ValueError(f'binary average requested for non-binary confusion matrix ({conf_mat.shape})')

        FP = conf_mat[0, 1]
        TP = conf_mat[1, 1]
        FN = conf_mat[1, 0]
        TN = conf_mat[0, 0]
    else:
        # multiclass case, see confusion matrix illustration
        # for multiclass cases in the links above
        FP = conf_mat.sum(axis=0) - diag
        FN = conf_mat.sum(axis=1) - diag
        TP = diag
        TN = all_sum - (FP + FN + TP)

    if average == 'micro':
        # globally sum every stat
        FP = FP.sum()
        FN = FN.sum()
        TP = TP.sum()
        TN = TN.sum()

    TPR = safe_div(TP, (TP + FN), NaN)
    FPR = safe_div(FP, (FP + TN), NaN)
    PRECISION = safe_div(TP, (TP + FP), NaN)

    if average == 'macro':
        # take the average of each stats
        TPR = np.average(TPR)
        FPR = np.average(FPR)
        PRECISION = np.average(PRECISION)

    # Calculate accuracy:
    # The same across all kinds of averages. It is the sum of the diagonal
    # of the confusion matrix divided by the sum of the entire matrix.
    # Split to cases because the diagonal sum is obtained differently
    # purely because the state of the variables.
    # Mathematically, it is the same calculation.
    if average == 'micro':
        ACC = TP / all_sum
    elif average == 'macro':
        ACC = TP.sum() / all_sum
    else:
        ACC = (TP + TN) / all_sum

    return TPR, FPR, ACC, PRECISION


def buildLayers(in_layer, dropout_rate=0.5, isDropout=False):
    """
    Builds the hidden layers of the neural network model.
    :param in_layer: The input layer of the model
    :param dropout_rate: The drop rate used for each dropout layer
    :param isDropout: Whether to use dropout layers between 2
    fully connected layers
    :return: The last hidden layer of the model
    """
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
    """
    Returns a factory for the keras classifier adapter
    :param method: The training algorithm kind
    :param input_dim: The dimension of the input layer (amount of attributes)
    :param classes_count: The total amount of classes
    :return:
    """
    def buildModel(epsilon=1e-3, alpha=1, dropout_rate=0.2):
        """
        Builds the neural network VAT model using the specified hyper-parameters,
        method, input dimension and classes count.

        For binary classification, only one neuron with
        sigmoid is used in the output layer and the loss is
        binary cross entropy.

        For multiclass classification, there are as many neurons in the output
        layer as there are classes, with softmax activation function and
        the loss is categorical cross entropy (requires one-hot label encoding).

        :param epsilon: epsilon hyper-parameter
        :param alpha: alpha hyper-parameter
        :param dropout_rate: The drop rate for each dropout layer (if applicable)
        :return: A newly compiled keras VAT model
        """

        # xi hyper-paramter
        xi = 1e-6

        # * See the description for the changes between
        #   binary and multiclass classification.
        # * base_loss: the 'fancy' l in the article or
        #   the whole loss if not using VAT
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

        # initialize the model
        model = VatKerasModel(
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
    """
    Measures the amount of time it takes the model to predict
    on 1000 instances from the dataset.
    :param X: The instances dataset without label information
    :param model: The model
    :return: The time the model took to predict 1000 instances from the dataset
    """

    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)

    # the indices used for inference
    selected_indexes = indexes[:1000]
    x_test = X[selected_indexes, :]

    start_time = time.time()
    model.predict(x_test)
    return time.time() - start_time


def start_evaluation(method, should_save):
    """
    Runs the evaluation on each dataset in the list
    :param method: The algorithm kind
    :param should_save: Whether saving the results to files is enabled
    :return:
    """
    setup()
    datasets_names = get_datasets_names()
    amount_of_datasets = len(datasets_names)
    for iteration, dataset_name in enumerate(datasets_names):
        evaluate(dataset_name, method, should_save)
        print(f'Done processing {iteration + 1} datasets from {amount_of_datasets}')


def evaluate(
        dataset_name,
        method,
        should_save,
        n_cv_outer_splits=CV_OUTER_N_ITERATIONS,
        n_cv_inner_splits=CV_INNER_N_ITERATIONS,
        epochs=EPOCHS,
        n_random_search_iters=N_RANDOM_SEARCH_ITERS
):
    """
    Evaluates the model on the specified dataset and algorithm kind
    and (maybe) saves the results to a file.

    First trains the model using nested cross-validation (both stratified)
    and random search for hyper-parameter optimization
    as the inner cross-validation.
    After the inner cross-validation, refits the model on the best found
    hyper-parameters (implicit by the library, no code addition is required).

    Then, evaluates it's prediction performance using
    various stats on the test set (the outer cross-validation test-set)
    and (maybe) saves the results to a file.

    :param dataset_name: The name of the dataset file
    :param method: The algorithm kind, see the documentation
    on the model class for more information
    :param should_save: Whether saving the performance results to a file is enabled
    :param n_cv_outer_splits: The amount of spits and iterations
    in the outer cross-validation
    :param n_cv_inner_splits: The amount of spits and iterations
    in the inner cross-validation
    :param epochs: The amount of epochs the model will do each fit
    in the inner cross-validation
    :param n_random_search_iters: The amount of iteration the random search
    hyper-parameter optiomization does in each inner cross-validation.
    Each such iteration performs a fit 1 time.
    :return:
    """

    data, labels, classes_count, input_dim = read_data(dataset_name)

    # Chose the hyper-parameters used for optimization in the
    # inner cross-validation based on the algorithm kind.
    if method == 'Dropout':
        distributions = dict(dropout_rate=uniform(loc=1e-6, scale=1 - 1e-6))
    else:
        distributions = dict(alpha=np.linspace(0, 2, 101),
                             epsilon=uniform(loc=1e-6, scale=2e-3))

    # get a factory of the keras model for the keras classifier adapter
    model_factory = configHyperModelFactory(method, input_dim, classes_count)
    outer_cv = StratifiedKFold(n_splits=n_cv_outer_splits)

    print(f'Working on: {dataset_name} with Algo: {method}')
    for iteration, (train_indexes, test_indexes) in enumerate(outer_cv.split(data, labels)):
        # outer cross-validation iteration

        performance = create_dict(dataset_name, method)

        # get the data for this iteration
        X_train, X_test = data[train_indexes, :], data[test_indexes, :]
        y_train, y_test = labels[train_indexes], labels[test_indexes]

        # initialize the keras classifier adapter
        model = KerasClassifierOur(
            num_classes=classes_count,
            build_fn=model_factory,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # initialize the inner cross-validation and hyper-parameter optimization
        clf = RandomizedSearchCV(
            model,
            param_distributions=distributions,
            n_iter=n_random_search_iters,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=n_cv_inner_splits),
            random_state=0
        )

        # get the current timestamp (for printing)
        fit_start_datetime = datetime.datetime.now()

        # print iteration information for run supervision
        print(
            f'Started iteration {iteration + 1}/{n_cv_outer_splits} at {fit_start_datetime:%H:%M:%S} '
            f'on dataset \'{dataset_name}\', algorithm variant \'{method}\'',
            end='', flush=True
        )

        # time the fit of the iteration
        fit_start_time = time.time()

        # run the inner cross-validation in it's entirety
        # (and refit using the best hyper-parameters)
        result = clf.fit(X_train, y_train)

        # print the amount of time the inner cross-validation took
        fit_time_delta = time.time() - fit_start_time
        print(f', time took: {format_timedelta(datetime.timedelta(seconds=fit_time_delta))}')

        # get the model trained on the best hyper-parameters (i.e. best model)
        # and use it to predict the test set
        best_model = result.best_estimator_
        y_predict = best_model.predict(X_test)
        y_predict_proba = best_model.predict_proba(X_test)

        # calculate various performance stats
        report = report_performance(data, y_predict, y_predict_proba, y_test, best_model, classes_count)

        if should_save:
            # stringify the hyper-parameters used for the performance evaluation
            if method == 'Dropout':
                hp_values = 'dropout_rate = ' + str(np.round(result.best_params_['dropout_rate'], 3))
            else:
                alpha_str = 'alpha = ' + str(np.round(result.best_params_['alpha'], 3))
                eps_str = f'epsilon = {np.round(result.best_params_["epsilon"] * 1000, 3)}e-3'
                hp_values = alpha_str + ', ' + eps_str

            # save the performance in the file
            save_to_dict(performance, iteration + 1, hp_values, *report)
            save_to_csv(performance, dataset_name + "_" + method)


def report_performance(dataset, y_predict, y_predict_proba, y_test, best_model, classes_count, is_print=False):
    """
    Calculates various performance stats on the model prediction (and actual labels).
    The following stats are calculated:
      - True positive rate
      - False positive rate
      - Accuracy
      - Precision
      - AUC ROC
      - Average precision score (AUC Precision-Recall)
      - Train time of the best model (already stored on the instance)
      - Inference time for 1000 instances

    :param dataset: The instances without label information
    :param y_predict: The predicted classes for the test set
    :param y_predict_proba: The predicted probability for each class on the test set
    :param y_test: The actual labels of the test set (sparse encoding, just numbers)
    :param best_model: The best model
    :param classes_count: The amount of actual classes the dataset has
    :param is_print: Whether to print the results to the screen
    :return: The various performance stats
    """

    # convert to one-hot encoding (not necessarily needed)
    y_test_one_hot = keras.utils.to_categorical(y_test)

    # Pick the average kind based the problem (binary or multiclass).
    # Also, for binary problems, take the probability predicted by the model
    # for each instance of it being a positive instance.
    if classes_count == 2:
        metrics_average = 'binary'
        y_positive_proba = y_predict_proba[:, 1]
    else:
        metrics_average = METRIC_AVERAGE

    # calculate stats from confusion matrix
    TPR, FPR, ACC, PRECISION = compute_tpr_fpr_acc(y_test, y_predict, labels=best_model.classes_,
                                                   average=metrics_average)

    # The AUC stats require different input based on the problem type
    # (i.e. binary or multiclass)
    if classes_count == 2:
        # if the test set use has only 1 class, AUC ROC cannot be calculated.
        if len(np.unique(y_test)) != 2:
            AUC_ROC = NaN
        else:
            # average cannot be binary here since it raises an exception
            AUC_ROC = roc_auc_score(y_test, y_positive_proba, average=METRIC_AVERAGE, multi_class='ovo')

        # if the test set does not have a positive instance,
        # average precision score cannot be calculated.
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


def setup_gpu(gpu_mem_limit):
    """
    Returns a virtual GPU device with the specified amount of maximum VRAM usage.
    :param gpu_mem_limit: The maximum amount of VRAM usage
    :return: A virtual GPU with the maximum VRAM contraint (in MiB)

    Note: It seems like tensorflow reserves another 1GiB of VRAM,
    so this function passes to tensorflow the requested maximum
    amount minus 1GiB (1024 MiB).
    Therefore, it is unkown what passing 1024 or lower might actually do.
    """
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
    """
    Parses the commandline arguments
    :return: The parsed arguments and the arguments parser
    """

    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('algovar', choices=['Article', 'OUR', 'Dropout'])
    args_parser.add_argument('-cpu', default=False, required=False, const=True, action='store_const')
    args_parser.add_argument(
        '-no-save', default=False, required=False,
        const=True, action='store_const', dest='no_save'
    )
    args_parser.add_argument('--gpu-mem-limit', type=int, default=None, required=False)
    args = args_parser.parse_args()
    return args, args_parser


def choose_device(args):
    """
    Chose device to run on based on the command line arguments.
    May be None if no special requirements were made.
    :param args: The command line arguments
    :return: The device to run on, may be None.
    """
    device = None
    if args.cpu:
        device = tf.device('/CPU:0')
    elif args.gpu_mem_limit is not None:
        device = setup_gpu(args.gpu_mem_limit)
    return device


def run_on_device(method, device, should_save):
    """
    Runs the evaluation of the algorithm kind on the specified device.
    :param method: The algorithm kind
    :param device: The tensorflow device
    :param should_save: Whether to save the results to a file.
    :return:
    """
    if device is None:
        start_evaluation(method, should_save)
    else:
        with device:
            start_evaluation(method, should_save)


def main():
    """
    The entry point.
    Parses command line arguments, sets up and start running.
    :return:
    """
    args, args_parser = parse_args()
    print(f'args:', args)
    device = choose_device(args)
    run_on_device(args.algovar, device, not args.no_save)


def format_timedelta(td):
    """
    Formats a datetime.timedelta instance to a string.
    :param td: A datetime.timedelta
    :return: The string representing the datetime.timedelta instance
    """
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
