import pandas as pd
import scipy.stats


def statistic_test(data_filename, metric):
    measurements = split_df_to_measurements(pd.read_excel(data_filename), metric)
    return scipy.stats.friedmanchisquare(*measurements)


def split_df_to_measurements(df, metric):
    """
    Splits the dataframe to lists, a list per algorithm, of the mean
    of the outer cross validation performance of the chosen metric
    of each dataset for that algorithm.
    :param df: The dataframe to extract information from
    :param metric: The chosen metric to take the mean of and return as measurement for the statistic test
    :return:

    Note: code adopted from https://jamesrledoux.com/code/group-by-aggregate-pandas
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
    statistic_test('Results.xlsx', 'Accuracy')
