import pandas as pd


def save_to_dict(dict, TPR, FPR, ACC, PRECISION, AUC_ROC, AUC_Precision_Recall, train_time, inference_time):
    dict['TPR'].append(TPR)
    dict['FPR'].append(FPR)
    dict['ACC'].append(ACC)
    dict['PRECISION'].append(PRECISION)
    dict['AUC_ROC'].append(AUC_ROC)
    dict['AUC_Precision_Recall'].append(AUC_Precision_Recall)
    dict['train_time'].append(train_time)
    dict['inference_time'].append(inference_time)


def create_dict(dataset_name, algorithm_name):
    output = dict()
    output['Dataset Name'] = dataset_name
    output['Algorithm Name'] = algorithm_name
    output['TPR'] = []
    output['FPR'] = []
    output['ACC'] = []
    output['PRECISION'] = []
    output['AUC_ROC'] = []
    output['AUC_Precision_Recall'] = []
    output['train_time'] = []
    output['inference_time'] = []
    return output


def save_to_csv(dict, filename):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(filename + '.csv', index=False)
