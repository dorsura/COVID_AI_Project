# External includes
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
from time import time
# Statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
# pmdarima library
from pmdarima.arima import auto_arima
# Internal includes
from DataTransformation import DataTransformation
from AlgoRunner import AlgoRunner
from Preprocess import utils

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'


def generate_daily_new_cases_csv(city_code):
    """
    Generate a csv containing daily new cases by date.
    :param city_code: City code of the desired city.
    :return: No return. The function will generate a csv file called
    'daily_new_cases_time_series_{city_code}' that will contain two columns - date and number of
    daily cases.
    """
    # read csv file
    cities_df = pd.read_csv('../../Resources/corona_city_table_ver_00155.csv')

    # filter columns and rows
    cities_df = cities_df[['Date', 'City_Code', 'Cumulative_verified_cases']]
    out_df = cities_df.loc[cities_df['City_Code'] == city_code]

    # get rid of fields containing a "<15" value and replace them with ascending sequence of
    # number from 1 to 14
    column = 'Cumulative_verified_cases'
    city = city_code
    count = cities_df.loc[(cities_df[column] == "<15") & (cities_df['City_Code'] ==
                                                          city), column].count()
    factor = count / 14
    # if factor is less than 1, put the mean value in all fields and call it a day..
    if 0 <= factor < 1:
        out_df.loc[(cities_df[column] == "<15"), column] = 7
    else:
        number_of_rows_for_each_value = math.floor(factor)
        counter = 0
        i = 1
        for j, row in out_df.iterrows():
            if row['City_Code'] == city and row[column] == "<15":
                out_df.at[j, column] = i
                counter += 1
                if counter == number_of_rows_for_each_value:
                    i += 1
                    counter = 0
                    if i == 15:
                        break

    print(f'processed column {column} for city {city}')
    out_df[column].replace({"<15": 14}, inplace=True)
    out_df[column] = pd.to_numeric(out_df[column])

    out_df.reset_index(inplace=True)

    # generate the daily new cases column
    out_df['daily_new_cases'] = \
        out_df['Cumulative_verified_cases'] - out_df.shift(periods=1)[
            'Cumulative_verified_cases']
    out_df = out_df.dropna()

    # drop redundant columns
    out_df.drop(columns=['index', 'City_Code', 'Cumulative_verified_cases'], inplace=True)

    # If you want to change the dates in the time series use the following code:
    # params = DataParams()
    # out_df['Date'] = pd.to_datetime(out_df['Date'])
    # out_df = out_df[(out_df['Date'] >= params.start_date) & (out_df['Date'] <= params.end_date)]

    utils.generate_output_csv(out_df, f'daily_new_cases_time_series_{city_code}')


class StationarityTests:
    def __init__(self, significance=.05):
        self.significance_level = significance
        self.p_value = None
        self.is_stationary = None

    def ADF_Stationarity_Test(self, time_series, printResults=True):
        # Dickey-Fuller test:
        adf_test = adfuller(time_series, autolag='AIC')

        self.p_value = adf_test[1]

        if self.p_value < self.significance_level:
            self.is_stationary = True
        else:
            self.is_stationary = False

        if printResults:
            df_results = pd.Series(adf_test[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used',
                                         '# Observations Used'])

            # Add Critical Values
            for key, value in adf_test[4].items():
                df_results['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(df_results)


def test_transformations():
    """
    This method tests the DataTransformation library. We do this by running each transformation,
    and then applying the inverse transformation to make sure we get back the original time
    series. We use a time series that shows the daily new cases in Tel Aviv.
    """
    print('Data transformation test start')
    # Generate daily new cases for Tel Aviv (city code 5000)
    time_series = generate_time_series_for_city(5000)
    start_date = datetime(2020, 3, 12)
    end_date = datetime(2021, 9, 5)
    time_series = time_series[(time_series.index >= start_date)
                              & (time_series.index <= end_date)]
    plot_time_series(time_series, 'COVID Daily new cases Tel Aviv', 'Daily new cases')
    transformer = DataTransformation(time_series.copy())
    s_test = StationarityTests()
    # Test for stationarity
    s_test.ADF_Stationarity_Test(time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # We see that the time series is NOT stationary, so let's start applying transformations

    # Difference transformation
    time_series_diff = transformer.difference(lags=1)
    print('Differenced time series:\n', time_series_diff.head())
    plot_time_series(time_series_diff, 'Diff(1) Transformation', 'Diff values')
    s_test.ADF_Stationarity_Test(time_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Invert diff
    transformer.invert_difference(time_series_diff, time_series.copy(), lags=1)

    # Square root transformation
    sqrt_time_series = transformer.sqrt()
    print('Sqrt time series:\n', sqrt_time_series.head())
    plot_time_series(sqrt_time_series, 'Square root Transformation', 'Square root values')
    s_test.ADF_Stationarity_Test(sqrt_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # take diff on top of sqrt transformation
    sqrt_diff_time_series = transformer.difference(1)
    plot_time_series(sqrt_diff_time_series, 'Square root and Diff(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(sqrt_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Invert transformations
    transformer.invert_difference(sqrt_diff_time_series, sqrt_time_series, lags=1)
    inverted = transformer.pow()
    print('Inverted time series:\n', inverted.head())
    # Print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Power transformation
    pow_transformation = transformer.pow()
    print('Power time series:\n', pow_transformation.head())
    plot_time_series(pow_transformation, 'Power of 2 Transformation', 'Power values')
    s_test.ADF_Stationarity_Test(pow_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Invert
    inverted = transformer.sqrt()
    print('Inverted time series:\n', inverted.head())
    # Print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Log transformation
    log_transformation = transformer.log(increment_val=1)
    print('ln time series:\n', log_transformation.head())
    plot_time_series(log_transformation, 'ln(1) Transformation', 'ln(1) values')
    s_test.ADF_Stationarity_Test(log_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Take diff on top of log transformation
    log_diff_time_series = transformer.difference(lags=1)
    plot_time_series(log_diff_time_series, 'ln(1) and Diff(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Invert transformations
    transformer.invert_difference(log_diff_time_series, log_transformation, lags=1)
    inverted = transformer.exp(1)
    print('Inverted time series:\n', inverted.head())
    # Print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Standardization transformation
    standardized_time_series = transformer.standardization()
    print('Standardized time series:\n', standardized_time_series.head())
    plot_time_series(standardized_time_series, 'Standardization Transformation',
                     'Standardized values')

    s_test.ADF_Stationarity_Test(standardized_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Take log on top of standardization transformation
    standardized_log_time_series = transformer.log(increment_val=1)
    plot_time_series(standardized_log_time_series, 'Standardization and ln(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(standardized_log_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Take diff on top of standardization and log transformation
    standardized_log_diff_time_series = transformer.difference(lags=1)
    plot_time_series(standardized_log_diff_time_series, 'Standardization, ln(1) and Diff(1) '
                                                        'Transformation', 'Transformation values')
    s_test.ADF_Stationarity_Test(standardized_log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Invert transformations
    transformer.invert_difference(standardized_log_diff_time_series,
                                  standardized_log_time_series, lags=1)
    transformer.exp(decrement_val=1)
    inverted = transformer.invert_standardization()
    print('Inverted time series:\n', inverted.head())
    plot_time_series(inverted, 'Inverted', 'Values')

    print('Data transformation test finished successfully!')


def plot_time_series(time_series, title=None, ylabel=None, end_date=None):
    """
    Plot a time series graph. Plotting plot lines from March-2020 until November-2021.
    :param time_series: Time series to plot.
    :param title: Plot title.
    :param ylabel: Y axis label.
    :param end_date: Date to end the plotting.
    """
    if end_date is not None:
        time_series = time_series[(time_series.index <= end_date)]
    plt.figure(figsize=(10, 4))
    plt.plot(time_series)
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=0.5, axis='x')
    plt.axhline(time_series.iloc[:, 0].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(time_series.iloc[:, 0].std(), color='b', alpha=0.2, linestyle='--')

    plt.show()


def generate_time_series_for_city(city_code):
    """
    A simple function that creates a time series for a given city.
    :param city_code: The city that we want to create this time series for.
    :return: Time series dataset.
    """
    generate_daily_new_cases_csv(city_code)
    time_series = pd.read_csv(f'daily_new_cases_time_series_{city_code}.csv', index_col=0,
                              parse_dates=True)
    time_series = time_series.asfreq(pd.infer_freq(time_series.index))

    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series


def generate_rolling_average_series():
    """
    A simple function that creates a time series that holds for each day the 7 day rolling
    average of new cases.
    :return: Time rolling average time series.
    """
    time_series = pd.read_csv('../../Resources/7_days_rolling_avg_global.csv', index_col=0)
    time_series.index = pd.to_datetime(time_series.index, format='%d-%m-%Y')
    start_date = datetime(2020, 2, 12)
    end_date = datetime(2021, 11, 22)
    time_series = time_series[(time_series.index >= start_date) & (time_series.index <=
                                                                   end_date)]

    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series


def generate_daily_cases_national():
    """
    A simple function that creates a time series that holds for each day the total number of new
    cases.
    :return: Time daily new cases time series.
    """
    time_series = pd.read_csv('../../Resources/daily_cases_global.csv', index_col=0)
    time_series.index = pd.to_datetime(time_series.index, format='%d-%m-%Y')
    start_date = datetime(2020, 2, 12)
    end_date = datetime(2021, 11, 22)
    time_series = time_series[(time_series.index >= start_date) & (time_series.index <=
                                                                     end_date)]
    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series


def print_time_series_for_ten_largest_cities():
    """
    Print a plot containing daily new cases for the ten largest cities.
    """
    top_ten_cities_dict = {
        3000: "Jerusalem",
        5000: "Tel Aviv",
        4000: "Haifa",
        8300: "Rishon Lezion",
        7900: "Petah Tikva",
        70: "Ashdod",
        7400: "Netanya",
        6100: "Bnei Brak",
        9000: "Be'er Sheva",
        6600: "Holon"
    }
    plt.figure(figsize=(10, 4))
    plt.grid(color='k', linestyle='--', linewidth=0.5, axis='x')
    plt.title('Daily new cases for most populated cities', fontsize=20)
    plt.ylabel('Daily new cases', fontsize=16)
    for key, value in top_ten_cities_dict.items():
        time_series = generate_time_series_for_city(key)
        plt.plot(time_series, label=value)
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize=12)
    plt.show()


def run_ma_model_demonstration():
    """
    Demonstrate how the MA model works.
    """
    print("--------Running MA model demonstration--------")
    national_daily_cases = generate_daily_cases_national()

    # Define the training set
    train_end = datetime(2020, 12, 20)
    train_series = national_daily_cases[(national_daily_cases.index <= train_end)]
    plot_time_series(train_series, 'National daily new cases', 'Daily new cases',
                     end_date=train_end)

    # Test for stationarity
    s_test = StationarityTests()
    s_test.ADF_Stationarity_Test(train_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # The time series is not stationary, so we'll apply transformations in order to make it
    # stationary
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    plot_time_series(time_series_transformed, 'National daily new cases (transformed)',
                     'Transformed values',
                     end_date=train_end)
    s_test.ADF_Stationarity_Test(time_series_transformed.copy(), printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Define the runner that will run our MA model.
    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # Print autocorrelation plot (ACF)
    runner.print_correlation_plots(number_of_lags=25, print_acf=True, print_pacf=False)
    # Run MA(2) and predict 2 days
    test_end = datetime(2020, 12, 22)
    runner.run_ma_regressor(ma_order=2, train_end=train_end, test_end=test_end)
    # Run MA(7) and predict 7 days
    test_end = datetime(2020, 12, 27)
    runner.run_ma_regressor(ma_order=7, train_end=train_end, test_end=test_end)
    print("--------Finished MA model demonstration--------")


def run_ar_model_demonstration():
    """
    Demonstrate how the AR model works.
    """
    print("--------Running AR model demonstration--------")
    national_daily_cases = generate_daily_cases_national()

    # Define training and test sets
    train_end = datetime(2020, 12, 20)
    test_end = datetime(2020, 12, 27)
    train_series = national_daily_cases[(national_daily_cases.index <= train_end)]
    plot_time_series(train_series, 'National daily new cases', 'Daily new cases',
                     end_date=train_end)

    # Apply transformations
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    plot_time_series(time_series_transformed, 'National daily new cases (transformed)',
                     'Transformed values',
                     end_date=train_end)

    # Define the runner that will run our AR model.
    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # Print partial autocorrelation plot (PACF)
    runner.print_correlation_plots(number_of_lags=25, print_acf=False, print_pacf=True)
    # Run AR(2) and predict 7 days
    runner.run_ar_regressor(ar_order=2, train_end=train_end, test_end=test_end)
    # Run AR(7) and predict 7 days
    runner.run_ar_regressor(ar_order=7, train_end=train_end, test_end=test_end)
    print("Finished AR model demonstration")


def run_arma_model_demonstration():
    """
    Demonstrate how the ARMA model works.
    """
    print("--------Running ARMA model demonstration--------")
    national_daily_cases = generate_daily_cases_national()

    # Define training and test sets
    train_end = datetime(2020, 12, 20)
    test_end = datetime(2020, 12, 30)
    train_series = national_daily_cases[(national_daily_cases.index <= train_end)]
    plot_time_series(train_series, 'National daily new cases', 'Daily new cases',
                     end_date=train_end)

    # Apply transformations
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    plot_time_series(time_series_transformed, 'National daily new cases (transformed)',
                     'Transformed values',
                     end_date=train_end)

    # Define the runner that will run our AR model. We chose p=7 and q=7 according to the ACF and
    # PACF plots we already analyzed when running the AR and MA models.
    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # Run ARMA(7,7) and predict 10 days
    runner.run_arma_regressor(ar_order=7, ma_order=7, train_end=train_end, test_end=test_end)
    print("--------Finished ARMA model demonstration--------")


def run_arima_model_demonstration():
    """
    Demonstrate how the ARIMA model works.
    """
    print("--------Running ARIMA model demonstration--------")
    national_daily_cases = generate_daily_cases_national()

    # Define training and test sets
    train_end = datetime(2020, 12, 5)
    test_end = datetime(2020, 12, 12)
    train_series = national_daily_cases[(national_daily_cases.index <= train_end)]
    plot_time_series(train_series, 'National daily new cases', 'Daily new cases',
                     end_date=train_end)

    # Run diff transformation to find a suitable d parameter for ARIMA model
    transformer = DataTransformation(national_daily_cases.copy())
    # Diff(1)
    train_series_diff = transformer.difference(1)
    plot_time_series(train_series_diff, 'Diff(1) Transformation', 'Diff values',
                     end_date=train_end)
    s_test = StationarityTests()
    print("Diff(1) Transformation:")
    s_test.ADF_Stationarity_Test(train_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    transformer.invert_difference(train_series_diff, national_daily_cases, lags=1)
    # Diff(2)
    train_series_diff = transformer.difference(2)
    plot_time_series(train_series_diff, 'Diff(2) Transformation', 'Diff values',
                     end_date=train_end)
    s_test = StationarityTests()
    print("Diff(2) Transformation:")
    s_test.ADF_Stationarity_Test(train_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Diff(2) results in a better P-value, so we will set d=2. for p and q we choose the same
    # parameters we chose for the ARMA model.
    runner = AlgoRunner(national_daily_cases)
    runner.run_arima_regressor(ar_order=7, diff_order=2, ma_order=7, train_end=train_end,
                               test_end=test_end)
    print("--------Finished ARIMA model demonstration--------")


def run_sarima_model_demonstration():
    """
    Demonstrate how the SARIMA model works.
    """
    print("--------Running SARIMA model demonstration--------")
    national_daily_cases = generate_daily_cases_national()

    # Define training and test sets
    train_end = datetime(2020, 12, 20)
    test_end = datetime(2020, 12, 30)
    train_series = national_daily_cases[(national_daily_cases.index <= train_end)]
    plot_time_series(train_series, 'National daily new cases', 'Daily new cases',
                     end_date=train_end)

    # Decompose the time series into the trend, seasonal and residual components.
    decompose_result_mult = seasonal_decompose(train_series, model="additive")
    # You can plot each component separately if you want
    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid
    # Plot all components
    decompose_result_mult.plot()
    plt.show()

    # Apply transformations
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    plot_time_series(time_series_transformed, 'National daily new cases (transformed)',
                     'Transformed values',
                     end_date=train_end)

    # Decompose the transformed time series.
    train_series = time_series_transformed[(time_series_transformed.index <= train_end)]
    decompose_result_mult = seasonal_decompose(train_series, model="additive")
    # Plot all components
    decompose_result_mult.plot()
    plt.show()

    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # Plot the correlation plots the remind us what are the appropriate lags for MA and AR.
    runner.print_correlation_plots(number_of_lags=25)
    # We choose p=2 and q=2 for the non-seasonal orders.
    model_orders = (2, 0, 2)
    # For the seasonal orders we choose P=1,D=1,Q=1 and M=7.
    seasonal_orders = (1, 1, 1, 7)
    runner.run_sarima_regressor(model_orders, seasonal_orders, train_end, test_end,
                                use_rolling_forecast=False)
    print("--------Finished SARIMA model demonstration--------")


def run_auto_arima_experiment():
    """
    Run the auto arima experiment in order to tune the models parameters.
    """
    print("--------Running auto ARIMA experiment--------")
    national_daily_cases = generate_daily_cases_national()
    plot_time_series(national_daily_cases, 'National daily new cases', 'Daily new cases')
    # Run the naive auto arima on the original time series
    start = time()
    model_auto = auto_arima(national_daily_cases)
    end = time()
    print(f'Auto ARIMA runtime: {end-start} seconds')
    print(model_auto.summary())

    # Apply transformations
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    plot_time_series(time_series_transformed, 'National daily new cases (transformed)',
                     'Transformed values')
    # Check for stationarity
    s_test = StationarityTests()
    s_test.ADF_Stationarity_Test(time_series_transformed, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # Run the naive auto arima on the transformed time series
    start = time()
    model_auto = auto_arima(time_series_transformed, stationary=True)
    end = time()
    print(f'Auto ARIMA runtime: {end-start} seconds')
    print(model_auto.summary())

    # Now set the parameters so that we try to get the best results for our model
    start = time()
    model_auto = auto_arima(time_series_transformed, max_p=7, max_q=7, max_P=4, max_Q=4, max_D=2,
                            max_order=None, stationary=True, seasonal=True, m=7, maxiter=50,
                            information_criterion='aic')
    end = time()
    print(f'Auto ARIMA runtime: {end-start} seconds')
    print(model_auto.summary())

    # Compare between the naive model and the model obtained by tweaking the parameters.
    train_end = datetime(2021, 7, 20)
    test_end = datetime(2021, 7, 30)
    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # First run the naive model ARIMA(3,0,1)
    naive_predictions = runner.run_arima_regressor(3, 0, 1, train_end, test_end)
    # Now run the optimized model SARIMA(3,0,5)(1,0,2,7)
    optimized_predictions = runner.run_sarima_regressor((3, 0, 5), (1, 0, 2, 7), train_end,
                                                        test_end)
    # Plot the original data vs. predictions of each model
    plt.figure(figsize=(10, 4))
    data = national_daily_cases
    data = data[(data.index <= test_end)]
    plt.plot(data)
    plt.plot(naive_predictions)
    plt.plot(optimized_predictions)
    plt.legend(('Data', 'ARIMA(3,0,1)', 'SARIMA(3,0,5)(1,0,2,7)'), fancybox=True, framealpha=1,
               shadow=True, borderpad=1, fontsize=12)
    plt.title('Data vs. Predictions', fontsize=20)
    plt.ylabel('Values', fontsize=16)
    plt.show()
    print("--------Finished auto ARIMA experiment--------")


def run_rolling_forecast_experiment():
    """
    Run the rolling forecast experiment.
    :return:
    """
    print("--------Running rolling forecast experiment--------")
    national_daily_cases = generate_daily_cases_national()
    plot_time_series(national_daily_cases, 'National daily new cases', 'Daily new cases')

    # Define training and test sets
    train_end = datetime(2021, 2, 1)
    test_end = datetime(2021, 3, 1)
    # First try to predict one month with the regular forecasting algorithm
    runner1 = AlgoRunner(national_daily_cases)
    runner1.run_arima_regressor(7, 2, 7, train_end, test_end, use_rolling_forecast=False)
    # Now try to predict the same month with the a rolling forecast
    runner1.run_arima_regressor(7, 2, 7, train_end, test_end, use_rolling_forecast=True)
    print("--------Finished rolling forecast experiment--------")


def run_rolling_average_smoothing_experiment():
    """
    Run the rolling average smoothing experiment. In this experiment we create the 7 day rolling
    average time series, and use it to predict the 3rd and 4th waves of COVID outbreaks.
    """
    print("--------Running rolling average smoothing experiment--------")
    national_daily_cases = generate_daily_cases_national()
    # Show the volatility using seasonal decompose
    decompose_result_mult = seasonal_decompose(national_daily_cases, model="additive")
    # You can plot each component separately if you want
    residual = decompose_result_mult.resid
    plt.plot(residual)
    plt.title('Residuals from daily new cases', fontsize=20)
    plt.ylabel('Resid', fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=0.5, axis='x')
    plt.show()
    # Create the rolling average series
    rolling_average_series = generate_rolling_average_series()
    # Plot daily new cases vs. 7 day rolling average
    plt.figure(figsize=(10, 4))
    plt.plot(national_daily_cases)
    plt.plot(rolling_average_series)
    plt.legend(('Daily new cases', '7 Day rolling average'), fancybox=True,
               framealpha=1, shadow=True, borderpad=1, fontsize=12)
    plt.title('Daily new cases vs. Rolling average', fontsize=20)
    plt.ylabel('Values', fontsize=16)
    plt.grid(color='k', linestyle='--', linewidth=0.5, axis='x')
    plt.show()

    plot_time_series(rolling_average_series, '7 Day rolling average', 'Average values')
    # Test for stationarity
    s_test = StationarityTests()
    s_test.ADF_Stationarity_Test(rolling_average_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Compare between the basic ARMA models using rolling forecast
    run_model_comparison_using_rolling_average()

    # Forecast the 3rd wave using basic ARIMA model and rolling forecast
    train_end = datetime(2020, 12, 1)
    test_end = datetime(2021, 4, 1)
    runner = AlgoRunner(rolling_average_series)
    runner.run_arima_regressor(2, 1, 2, train_end, test_end, use_rolling_forecast=True)

    # Forecast the 4th wave using basic ARIMA model and rolling forecast
    train_end = datetime(2021, 7, 1)
    test_end = datetime(2021, 10, 1)
    runner.run_arima_regressor(2, 1, 2, train_end, test_end, use_rolling_forecast=True)
    print("--------Finished rolling average smoothing experiment--------")


def compare_all_models():
    """
    Forecast a certain time period with all models and compare the results.
    """
    national_daily_cases = generate_daily_cases_national()
    # Apply transformations
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    runner = AlgoRunner(time_series_transformed, original_time_series=national_daily_cases,
                        transformations=['log', 'difference'],
                        diff_params=(1, time_series_log), log_exp_delta=1)
    # Define training and test sets. We will make a 10 day forecast
    train_end = datetime(2020, 12, 20)
    test_end = datetime(2020, 12, 30)
    ma_pred = runner.run_ma_regressor(7, train_end, test_end, use_rolling_forecast=False,
                                      print_results=False)
    runner.print_result_metrics('MA')
    ar_pred = runner.run_ar_regressor(7, train_end, test_end, use_rolling_forecast=False,
                                      print_results=False)
    runner.print_result_metrics('AR')
    arma_pred = runner.run_arma_regressor(7, 7, train_end, test_end, use_rolling_forecast=False,
                                          print_results=False)
    runner.print_result_metrics('ARMA')
    arima_pred = runner.run_arima_regressor(7, 2, 7, train_end, test_end,
                                            use_rolling_forecast=False, print_results=False)
    runner.print_result_metrics('ARIMA')
    sarima_pred = runner.run_sarima_regressor((2, 0, 2), (1, 1, 1, 7), train_end, test_end,
                                              use_rolling_forecast=False, print_results=False)
    runner.print_result_metrics('SARIMA')

    # Plot the original data vs. predictions of each model
    plt.figure(figsize=(10, 4))
    data = national_daily_cases
    plot_start_date = datetime(2020, 12, 1)
    data = data[(data.index <= test_end)]
    data = data[(data.index >= plot_start_date)]
    plt.plot(data)
    plt.plot(ma_pred)
    plt.plot(ar_pred)
    plt.plot(arma_pred)
    plt.plot(arima_pred)
    plt.plot(sarima_pred)
    plt.legend(('Data', 'MA(7)', 'AR(7)', 'ARMA(7,7)', 'ARIMA(7,2,7)', 'SARIMA(2,0,2)(1,1,1,7)'),
               fancybox=True,
               framealpha=1,
               shadow=True, borderpad=1, fontsize=12)
    plt.title('Data vs. Predictions', fontsize=20)
    plt.ylabel('Values', fontsize=16)
    plt.show()


def run_model_comparison_using_rolling_average():
    """
    Forecast a certain time period with several models and compare the results. This experiment
    uses the 7 day rolling average time series.
    """
    rolling_average_series = generate_rolling_average_series()
    train_end = datetime(2020, 12, 1)
    test_end = datetime(2021, 1, 1)
    # Apply transformations
    transformer = DataTransformation(rolling_average_series.copy())
    time_series_transformed = transformer.log(increment_val=1)
    runner = AlgoRunner(time_series_transformed, original_time_series=rolling_average_series,
                        transformations=['log'], log_exp_delta=1)
    ma_pred = runner.run_ma_regressor(4, train_end, test_end, use_rolling_forecast=True,
                                      print_results=False)
    runner.print_result_metrics('MA')
    ar_pred = runner.run_ar_regressor(4, train_end, test_end, use_rolling_forecast=True,
                                      print_results=False)
    runner.print_result_metrics('AR')
    arma_pred = runner.run_arma_regressor(4, 4, train_end, test_end, use_rolling_forecast=True,
                                          print_results=False)
    runner.print_result_metrics('ARMA')
    # Plot the original data vs. predictions of each model
    plt.figure(figsize=(10, 4))
    data = rolling_average_series
    plot_start_date = datetime(2020, 11, 1)
    data = data[(data.index <= test_end)]
    data = data[(data.index >= plot_start_date)]
    plt.plot(data)
    plt.plot(ma_pred)
    plt.plot(ar_pred)
    plt.plot(arma_pred)
    plt.legend(('Data', 'MA(4)', 'AR(4)', 'ARMA(4,4)'),
               fancybox=True,
               framealpha=1,
               shadow=True, borderpad=1, fontsize=12)
    plt.title('Data vs. Predictions', fontsize=20)
    plt.ylabel('Values', fontsize=16)
    plt.show()


if __name__ == "__main__":
    # You can run each part individually or all together
    test_transformations()
    run_ma_model_demonstration()
    run_ar_model_demonstration()
    run_arma_model_demonstration()
    run_arima_model_demonstration()
    run_sarima_model_demonstration()
    compare_all_models()
    run_auto_arima_experiment()
    run_rolling_forecast_experiment()
    run_rolling_average_smoothing_experiment()








