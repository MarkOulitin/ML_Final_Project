import pandas as pd
import os
from pathlib import Path
import numpy as np


def save_to_dict(dict, iteration, hp_values, TPR, FPR, ACC, PRECISION, AUC_ROC, AUC_Precision_Recall, train_time,
                 inference_time):
    dict['Hyper-Parameters Values'].append(hp_values)
    dict['Cross Validation [1-10]'].append(iteration)
    dict['TPR'].append(np.round(TPR, 3))
    dict['FPR'].append(np.round(FPR, 3))
    dict['Accuracy'].append(np.round(ACC, 3))
    dict['Precision'].append(np.round(PRECISION, 3))
    dict['AUC ROC'].append(np.round(AUC_ROC, 3))
    dict['AUC Precision Recall'].append(np.round(AUC_Precision_Recall, 3))
    dict['Training Time'].append(np.round(train_time, 3))
    dict['Inference Time'].append(np.round(inference_time, 3))


def create_dict(dataset_name, algorithm_name):
    output = dict()
    output['Dataset Name'] = dataset_name
    output['Algorithm Name'] = algorithm_name
    output['Cross Validation [1-10]'] = []
    output['Hyper-Parameters Values'] = []
    output['TPR'] = []
    output['FPR'] = []
    output['Accuracy'] = []
    output['Precision'] = []
    output['AUC ROC'] = []
    output['AUC Precision Recall'] = []
    output['Training Time'] = []
    output['Inference Time'] = []
    return output


results_dir = 'Results'


def save_to_csv(dict, filename):
    global results_dir
    df = pd.DataFrame.from_dict(dict)
    file_path = results_dir + '/' + filename + '.csv'
    result_file = Path(file_path)
    if result_file.is_file():
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, index=False)


def setup():
    global results_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


def merge_results(result_filename, results_dir):
    list_of_results = []
    for filename in os.listdir(results_dir):
        list_of_results.append(pd.read_csv(results_dir + "/" + filename))

    pd.concat(list_of_results).to_excel(result_filename, index=False)


if __name__ == '__main__':
    merge_results('Results.xlsx', 'Final_Results')
