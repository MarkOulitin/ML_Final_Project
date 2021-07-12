import numpy as np
import pandas as pd
import scipy.stats
import scikit_posthocs

CONFIDENCE_ALPHA = 0.05


def statistic_test(data_filename, metric, alpha):
    """
    Performs friedman statistic test with confidence :alpha:
    on the performance the specified :metric: of the algorithms on the datasets.

    Performs a post-hoc nemenyi test if found significant and
    saves it's results to the specified file.

    Uses the mean of the specified metric of each algorithm on the datasets as
    the measurement for the friedman test.

    :param data_filename: The file name to read the algorithms performances on
    the various datasets
    :param metric: The metric to use for the statistic test
    :param alpha: The confidence value used to determine whether the test is
    significant
    :return: The results of the post-hoc test if significant, otherwise None.
    """
    measurements = split_df_to_measurements(pd.read_excel(data_filename), metric)
    statistic, pvalue = scipy.stats.friedmanchisquare(*measurements)
    if pvalue < alpha:
        return scikit_posthocs.posthoc_nemenyi_friedman(measurements_to_posthoc(measurements))
    else:
        return None


def measurements_to_posthoc(measurements):
    """
    Converts the measurements for the friedman test
    to an input for the post-hoc test.
    :param measurements: The measurements of the friedman test
    :return: The converted measurements
    """
    # code adopted from https://www.statology.org/nemenyi-test-python/
    a = np.array(*measurements)
    a = np.transpose(a)
    return a


def split_df_to_measurements(df, metric):
    """
    Splits the dataframe to lists, a list per algorithm, of the mean
    of the outer cross validation performance of the chosen metric
    of each dataset for that algorithm.
    :param df: The dataframe to extract information from
    :param metric: The chosen metric to take the mean of and return as measurement for the statistic test
    :return:

    Note: code adopted from
        https://jamesrledoux.com/code/group-by-aggregate-pandas
        https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
    """
    metric_col_name = f'{metric}_mean'

    # calculate the mean of the metric for each algorithm and dataset pair
    df = df.groupby(['Algorithm Name', 'Dataset Name']).agg({metric: ['mean']})
    df.columns = [metric_col_name]

    # remove the grouping, flatten to a normal table so we can later
    # group by only the algorithm
    df = df.reset_index()

    df = df.groupby(['Algorithm Name'])

    # split to the lists of the means per algorithm
    df = df[metric_col_name].apply(list)
    return df.values


if __name__ == '__main__':
    statistic_test('Results.xlsx', 'Accuracy', 0.05)
