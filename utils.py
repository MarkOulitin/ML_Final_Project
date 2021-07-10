import pandas as pd
import os


def save_to_dict(dict, iteration, hp_values, TPR, FPR, ACC, PRECISION, AUC_ROC, AUC_Precision_Recall, train_time, inference_time):
    dict['Hyper-Parameters Values'].append(hp_values)
    dict['Cross Validation [1-10]'].append(iteration)
    dict['TPR'].append(TPR)
    dict['FPR'].append(FPR)
    dict['Accuracy'].append(ACC)
    dict['Precision'].append(PRECISION)
    dict['AUC ROC'].append(AUC_ROC)
    dict['AUC Precision Recall'].append(AUC_Precision_Recall)
    dict['Training Time'].append(train_time)
    dict['Inference Time'].append(inference_time)


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
    df.to_csv(results_dir + '/' + filename + '.csv', index=False)


def setup():
    global results_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


def merge_results(result_filename):
    global results_dir

    list_of_results = []
    for filename in os.listdir(results_dir):
        list_of_results.append(pd.read_csv(results_dir + "/" + filename))

    pd.concat(list_of_results).to_excel(result_filename, index=False)


if __name__ == '__main__':
    merge_results('Results.xlsx')
