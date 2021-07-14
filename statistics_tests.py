import numpy as np
import pandas as pd
import scikit_posthocs
import scipy.stats
import utils

CONFIDENCE_ALPHA = 0.05


def statistic_test(df, metric, alpha):
    """
    Performs friedman statistic test with confidence :alpha:
    on the performance the specified :metric: of the algorithms on the datasets.

    Performs a post-hoc nemenyi test if found significant and
    saves it's results to the specified file.

    Uses the mean of the specified metric of each algorithm on the datasets as
    the measurement for the friedman test.

    :param df: The data frame from which to read the algorithms performances on
    the various datasets
    :param metric: The metric to use for the statistic test
    :param alpha: The confidence value used to determine whether the test is
    significant
    :return: The results of the post-hoc test if significant, otherwise None.
    """
    algorithms_measurements_pairs_list = split_df_to_measurements(df, metric)
    measurements = list(map(lambda p: p[1], algorithms_measurements_pairs_list))
    statistic, pvalue = scipy.stats.friedmanchisquare(*measurements)
    posthoc_results = None
    if pvalue < alpha:
        posthoc_measurements = measurements_to_posthoc(measurements)
        posthoc_results = scikit_posthocs.posthoc_nemenyi_friedman(posthoc_measurements)
        posthoc_results = posthoc_results.rename(
            index=algorithm_indices_to_dict(algorithms_measurements_pairs_list, posthoc_results.index),
            columns=algorithm_indices_to_dict(algorithms_measurements_pairs_list, posthoc_results.columns)
        )

    return pvalue, posthoc_results


def algorithm_indices_to_dict(algorithms_measurements_pairs_list, indices):
    return dict(map(lambda p: (p, algorithms_measurements_pairs_list[p][0]), indices))


def measurements_to_posthoc(measurements):
    """
    Converts the measurements for the friedman test
    to an input for the post-hoc test.
    :param measurements: The measurements of the friedman test
    :return: The converted measurements
    """
    # code adopted from https://www.statology.org/nemenyi-test-python/
    a = np.array(measurements)
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
    algorithms_measurements_pairs_list = list(df.to_dict().items())
    return algorithms_measurements_pairs_list


def run_statistic_test_from_results(results_file_name, metric, alpha, results_dir, posthoc_results_file_name):
    utils.merge_results(results_file_name, results_dir=results_dir)
    df = pd.read_excel(results_file_name)
    pvalue, posthoc_results = statistic_test(df, metric, alpha)
    if posthoc_results is None:
        print(f'Results are not significant alpha={alpha}, pvalue={pvalue}')
    else:
        print(f'Results are significant alpha={alpha}, pvalue={pvalue}')
        print(posthoc_results)
        posthoc_results.to_excel(posthoc_results_file_name)


if __name__ == '__main__':
    run_statistic_test_from_results(
        'Results.xlsx',
        metric='TPR',
        alpha=0.05,
        results_dir='Final_Results',
        posthoc_results_file_name='Posthoc_Results.xlsx'
    )
