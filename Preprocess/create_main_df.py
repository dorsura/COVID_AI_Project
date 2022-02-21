import pandas as pd
from Preprocess import utils
from Preprocess import extract_daily_new_cases
from Preprocess import extract_vaccination_stats
from Preprocess import normalize_df


def shift_columns(df):
    """
    Shift up by 1 all relevant columns from the corona df. The relevant columns are those who
    contain data up until the current date (including the actual date that we want to predict).
    By shifting up by one we make each feature represent the value of the previous day,
    which fits our prediction goals.
    :param df: The corona data frame containing the columns that we want to shift.
    :return: The shifted data frame.
    """
    columns_to_shift = ['Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths',
                        'Cumulated_number_of_tests', 'Cumulated_number_of_diagnostic_tests',
                        'colour', 'final_score']
    for column in columns_to_shift:
        df[column] = df[column].shift(-1)
    return df


'''
This code generates the main corona data frame. We use external code that generates all columns 
and then aggregate all of the data. 
The only manual preprocessing we do is changing the color values to numbers. We do this manually 
since these values are written in Hebrew and it is a pain to replace it in the code.
Note that we join all data frames together based on the following keys: [City_Name,City_Code,
Date]. Therefore if you add another external data frame make sure it has exactly those keys to 
assure that the join works correctly.
'''

if __name__ == "__main__":
    # read main data frame (we assume it exists - if not, run utils.py to create it)
    corona_df = pd.read_csv('corona_city_table_preprocessed.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    params = utils.DataParams()

    # shift by 1 all columns that contain data on the current day. We want each feature to
    # represent information up to the current date (not including).
    corona_df = shift_columns(corona_df)
    # get external columns
    vaccination_df = extract_vaccination_stats.generate_vaccination_columns()
    daily_new_cases_df = extract_daily_new_cases.generate_daily_new_cases_df()

    # reduce to the dates set by DataParams
    start_date = params.start_date
    end_date = params.end_date
    corona_df = corona_df[(corona_df['Date'] >= start_date) & (corona_df['Date'] <= end_date)]

    # join all data frames together. The key for the join op is [City_Name,City_Code,Date]
    temp_df = pd.merge(corona_df, vaccination_df, how="inner", on=["City_Name", "City_Code",
                                                                   "Date"])
    merged_df = pd.merge(temp_df, daily_new_cases_df, how="inner", on=["City_Name", "City_Code",
                                                                       "Date"])

    result_df = normalize_df.normalize_data_set(merged_df)

    # generate the output df
    utils.generate_output_csv(result_df, 'corona_df')

    # generate train and test set according to the split date
    train_df = result_df[result_df['Date'] < params.split_date]
    test_df = result_df[result_df['Date'] >= params.split_date]

    utils.generate_output_csv(train_df, 'train_df')
    utils.generate_output_csv(test_df, 'test_df')
