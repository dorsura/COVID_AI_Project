from datetime import datetime
import pandas as pd
import math
import numpy as np
from sklearn import metrics


'''
This file contains parameters that are used across all of our data frames. We define them here to 
make sure we use the same params. 
'''


class DataParams:
    def __init__(self):
        self.start_date = datetime(2021, 1, 5)
        self.end_date = datetime(2021, 9, 25)
        self.split_date = datetime(2021, 9, 1)
        self.end_date_for_dt = datetime(2021, 11, 22)
        self.split_date_for_dt = datetime(2021, 11, 18)
        self.number_of_weeks_for_vaccination_stats = 2
        self.number_of_days_for_infected_stats = 14
        self.normalization_factor = 1
        self.not_normalized_columns = ['City_Name', 'City_Code', 'Date', 'colour', 'final_score',
                                       'today_verified_cases', 'rolling_average_7_days']
        self.Y = 'today_verified_cases'
        # self.Y = 'rolling_average_7_days'
        self.split_test_size = 0.2
        self.split_random_state = 1


def generate_output_csv(df, output_name):
    """
   Parameters
   ----------
   df : dataframe object
       the dataframe we want to convert to csv
   output_name : str
       csv file name (no need to add .csv suffix)
    """
    df.to_csv(f'{output_name}.csv', index=False, encoding='utf-8-sig')


def preprocess_raw_dataset():
    """
    This method preprocesses the raw dataset and makes it ready to be used by our algorithms. The
    method will save the processed data as csv named 'corona_city_table_preprocessed'
    """
    # read main data frame
    corona_df = pd.read_csv('../Resources/corona_city_table_ver_00155.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])

    # reduce to selected dates. These dates may be more than what we use in practice but this
    # generates extra data for future use if we need it
    start_date = datetime(2020, 8, 31)
    end_date = datetime(2021, 11, 23)
    corona_df = corona_df[(corona_df['Date'] >= start_date) & (corona_df['Date'] <= end_date)]

    city_codes = list(dict.fromkeys(corona_df['City_Code'].values))
    # get rid of fields containing a "<15" value and replace them with ascending sequence of
    # number from 1 to 14
    for column in corona_df.filter(regex="Cumulative_.*|Cumulated_.*"):
        for city in city_codes:
            count = corona_df.loc[(corona_df[column] == "<15") & (corona_df['City_Code'] ==
                                                                  city), column].count()
            factor = count / 14
            # if factor is less than 1, put the mean value in all fields and call it a day..
            if 0 <= factor < 1:
                corona_df.loc[(corona_df[column] == "<15") & (corona_df['City_Code'] ==
                                                              city), column] = 7
            else:
                number_of_rows_for_each_value = math.floor(factor)
                counter = 0
                i = 1
                for j, row in corona_df.iterrows():
                    if row['City_Code'] == city and row[column] == "<15":
                        corona_df.at[j, column] = i
                        counter += 1
                        if counter == number_of_rows_for_each_value:
                            i += 1
                            counter = 0
                            if i == 15:
                                break
            print(f'processed column {column} for city {city}')
        corona_df[column].replace({"<15": 14}, inplace=True)
        corona_df[column] = pd.to_numeric(corona_df[column])

    # generate the output df
    generate_output_csv(corona_df, 'corona_city_table_preprocessed')


def print_result_metrics(y_true, y_pred, one_liner=False):
    """
    Print model accuracy using various metrics.
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :param one_liner: Whether to print a compact version of the result metrics.
    """
    test_mean = np.mean(y_true.iloc[:, 0])
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    med = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    if not one_liner:
        print('###----Metrics for model accuracy---###')
        print('Test Data Mean: ', test_mean)
        print('Mean Absolute Error', mae)
        print('Mean Squared Error:', mse)
        print('Mean Absolute Percentage Error: ', mape)
        print('Root Mean Squared Error:', np.sqrt(mse))
        print('Median Absolute Error:', med)
        print('R2 score:', r2)
    else:
        print(f'test mean: {test_mean}, mape:{mape}, mae {mae}, mse:{mse}, mape:{mape}, rmse:{np.sqrt(mse)}, med:{med}, R2 score:{r2}, ')


if __name__ == "__main__":
    preprocess_raw_dataset()

