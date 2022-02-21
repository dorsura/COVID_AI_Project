import pandas as pd
from Preprocess.utils import DataParams
'''
right now every first row (first day) on each city has wrong value because of the shifts.
this bug is irrelevant if we dont need the first few days - so i think its ok
if we want to calculate the first few days good, need to work on each city seperately or fix each first row with special
treatment.
for each row(for city_code in cities_df['City_Code'].unique())    
'''


def generate_daily_new_cases_df():
    cities_df = pd.read_csv('corona_city_table_preprocessed.csv')
    cities_df['Date'] = pd.to_datetime(cities_df['Date'])
    params = DataParams()

    # Generate N columns of previous days new cases
    N = params.number_of_days_for_infected_stats
    for i in range(1, N + 2):
        cities_df[f'tmp_{i}'] = cities_df['Cumulative_verified_cases'].shift(periods=i)
        if i == 1:
            cities_df[f'verified_cases_{i-1}_days_ago'] = cities_df['Cumulative_verified_cases'] - cities_df[f'tmp_{i}']
        else:
            cities_df[f'verified_cases_{i-1}_days_ago'] = cities_df[f'tmp_{i-1}'] - cities_df[f'tmp_{i}']
            cities_df.drop([f'tmp_{i-1}'], axis=1, inplace=True)

    cities_df.rename(columns={'verified_cases_0_days_ago': 'today_verified_cases'}, inplace=True)

    result_columns = ['City_Name', 'City_Code', 'Date', 'today_verified_cases', 'rolling_average_7_days']
    for i in range(2, N + 2):
        result_columns.append(f'verified_cases_{i - 1}_days_ago')

    cities_df['rolling_average_7_days'] = cities_df['today_verified_cases'].rolling(window=7).mean()

    # set start and end date (see the constraints for start date in the file prolog)
    start_date = params.start_date
    end_date = params.end_date
    result_df = cities_df[(cities_df['Date'] >= start_date) & (cities_df['Date'] <= end_date)][result_columns]
    return result_df

