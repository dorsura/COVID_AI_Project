from random import sample
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from Preprocess.utils import DataParams

K = 40

params = DataParams()
cols = ['City_Name', 'City_Code', 'Date', 'Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths',
        'Cumulated_number_of_tests', 'Cumulated_number_of_diagnostic_tests', 'colour', 'final_score', 'vaccinated_dose_1_total',
        'vaccinated_dose_2_total', 'vaccinated_dose_3_total', 'dose_1_in_last_1_week', 'dose_1_in_last_2_week', 'dose_2_in_last_1_week',
        'dose_2_in_last_2_week', 'dose_3_in_last_1_week', 'dose_3_in_last_2_week', 'today_verified_cases', 'rolling_average_7_days',
        'verified_cases_1_days_ago', 'verified_cases_2_days_ago', 'verified_cases_3_days_ago', 'verified_cases_4_days_ago',
        'verified_cases_5_days_ago', 'verified_cases_6_days_ago', 'verified_cases_7_days_ago', 'verified_cases_8_days_ago',
        'verified_cases_9_days_ago', 'verified_cases_10_days_ago', 'verified_cases_11_days_ago',
        'verified_cases_12_days_ago', 'verified_cases_13_days_ago', 'verified_cases_14_days_ago']

best_columns = ['vaccinated_dose_3_total', 'dose_3_in_last_2_week', 'verified_cases_7_days_ago', 'City_Code',
                'verified_cases_14_days_ago', 'colour'] + [params.Y]

tlv_code = 5000
haifa_code = 4000
min = '2021-01-20'
max = '2021-09-11'


def play_knn():
    """
    This function was used at the start of the project to play with the basic features of the knn algorithem.
    it helped to test the filter of panda df, and the train and predict of Scikit-learn knn.
    :return: None
    """
    data = pd.read_csv('../Preprocess/output.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    haifa_tlv_data = data[data['City_Code'].isin([5000, 4000])]

    start_date = datetime(2021, 7, 20)
    end_date = datetime(2021, 7, 28)

    # we work in specified date range
    haifa_tlv_data = haifa_tlv_data[(haifa_tlv_data['Date'] > start_date) & (haifa_tlv_data['Date'] < end_date)]
    with_answers = haifa_tlv_data.copy()

    neigh = KNeighborsRegressor(n_neighbors=3)
    Y = haifa_tlv_data['today_verified_cases']
    haifa_tlv_data.drop(['today_verified_cases', 'City_Name', 'Date'], axis=1, inplace=True)
    X = haifa_tlv_data

    neigh.fit(X, Y)

    x_hat = [4000, 16917, 31, 20, 18, 6, 15, 16, 16, 9, 11, 14, 2, 9, 2, 8]
    y_pred = neigh.predict([x_hat])
    print(y_pred)


def run_knn_with_split(k, examples, weights='uniform'):
    """
    This function does Cross validation to the knn algorithm. it does it with TimeSeriesSplit that always uses the past
    to predict the future (in cross validation form) more on that in the dry part.
    :param k: k value of knn
    :param examples: examples for the cross validation
    :param weights: weights function of knn
    :return: None. just prints the accuracies
    """
    tscv = TimeSeriesSplit(n_splits=10)
    sum = 0
    for train_index, test_index in tscv.split(examples):
        train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]
        knn_output = run_knn(k, train_examples, test_examples, weights=weights)
        print_results(knn_output)
        sum += knn_output['acc']

    avg_accuracy = sum / tscv.n_splits
    print(f'avg accuracy: {avg_accuracy}')


def run_knn(k, train, test, weights='uniform'):
    """
    This function train knn on the train set and run it on the test set.
    :param k: k value of knn
    :param train: train set for knn
    :param test: test set for knn
    :param weights: weights function of knn
    :return: dict with accuracies and stats of the knn test
    """
    neigh = KNeighborsRegressor(n_neighbors=k, weights=weights)

    y_train = train[params.Y].values
    x_train = train.loc[:, train.columns != params.Y]
    neigh.fit(x_train.values, y_train)

    y_test = test[params.Y].values
    x_test = test.loc[:, test.columns != params.Y]
    acc = neigh.score(x_test.values, y_test)
    y_predicted = neigh.predict(x_test.values)
    mae = mean_absolute_error(y_test, y_predicted)

    return {'acc': acc, 'mae': mae, 'test_mean': y_test.mean(), 'y_test': y_test, 'y_predicted': y_predicted}


def plot_diff_graph(diff, x_label, y_label, title, y_mean):
    """
    This function plots the diff graphs
    :param diff: diff between y_test and y_pred
    :param x_label: x axis name
    :param y_label: y axis name
    :param title: graph title
    :param y_mean: y_mean of the test set
    :return: None
    """
    plt.plot(diff.sample(100, ignore_index=True), label="diff")
    plt.axhline(y_mean, color='r', alpha=0.2, linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_graphs(k, train, test, weights='uniform'):
    """
    This function plots some graphs of y_pred in respect to y_test.
    We plot the diff of the first 100 examples, of some ranges of y_test etc.
    :param k: k value of knn
    :param train: train set for knn
    :param test: test set for knn
    :param weights: weights function of knn
    :return: None
    """
    train = train[best_columns]
    test = test[best_columns]

    knn_output = run_knn(k, train, test, weights)
    y_test = knn_output['y_test']
    y_predicted = knn_output['y_predicted']
    plt.plot(y_test[:100], '.', label="y_true")
    plt.plot(y_predicted[:100], '.', label="y_predicted")
    plt.axhline(y_predicted[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_predicted[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('first 100 examples')
    plt.show()

    diff = abs(y_test - y_predicted)
    plt.errorbar(range(100), y_test[:100], diff[:100], linestyle='None', marker='^')
    plt.title('errorbar')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(diff[:100], label='error')
    plt.axhline(y_test[:100].mean(), color='r', alpha=0.2, linestyle='--', label='MEAN')
    plt.axhline(y_test[:100].std(), color='b', alpha=0.2, linestyle='--', label='STD')
    plt.xlabel('#Sample')
    plt.ylabel('Error')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('Residuals Of KNN')
    plt.show()

    y_no_patients = y_test[y_test == 0]
    y_pred_no_patients = y_predicted[y_test == 0]
    plt.plot(y_no_patients[:100], '.', label="y_true")
    plt.plot(y_pred_no_patients[:100], '.', label="y_predicted")
    plt.axhline(y_no_patients[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_no_patients[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('y_true equals 0')
    plt.show()

    y_spec_patients = y_test[(y_test >= 50) & (y_test <= 200)]
    y_pred_spec_patients = y_predicted[(y_test >= 50) & (y_test <= 200)]

    plt.plot(y_spec_patients[:100], '.', label="y_true")
    plt.plot(y_pred_spec_patients[:100], '.', label="y_predicted")
    plt.axhline(y_spec_patients[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_spec_patients[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('y_true 50 to 200')
    plt.show()


def print_ranges(train_lower, train_upper, test_upper):
    """
    This function prints the date range of the train and test set.
    :param train_lower: train min date
    :param train_upper: train max date
    :param test_upper: test max date
    :return:
    """
    train_lower = train_lower.strftime('%d-%m-%Y')
    train_upper = train_upper.strftime('%d-%m-%Y')
    test_upper = test_upper.strftime('%d-%m-%Y')
    print(f'Train date range: {train_lower} - {train_upper}. test: {train_upper} - {test_upper}')


def experiment_rolling_train(k, train, weights='uniform'):
    """
    This is an implementation of cross validation we tried.
    it is a rolling window, that each time we train on 30 days and test in the next 30 days.
    :param k: k value of knn
    :param train: train set to do the rolling train on
    :param weights: weights function of knn
    :return: None.
    """
    train = train[best_columns + ['Date']]
    train_lower = train['Date'].min()
    max_date = train['Date'].max()
    train_upper = train_lower + timedelta(days=30)
    test_upper = train_upper + timedelta(days=30)
    while test_upper < max_date:
        tmp_train = train[(train['Date'] > train_lower) & (train['Date'] <= train_upper)]
        tmp_test = train[(train['Date'] > train_upper) & (train['Date'] < test_upper)]
        knn_output = run_knn(k, tmp_train[best_columns], tmp_test[best_columns], weights)
        print_ranges(train_lower, train_upper, test_upper)
        print_results(knn_output)

        train_lower = train_upper
        train_upper = test_upper
        test_upper = train_upper + timedelta(days=30)


def experiment_features(examples, columns):
    """
    This function move over a range of m and test the knn algorithm with m features.
    :param examples: examples to run the train on.
    :param columns: features list to test
    :return: None
    """
    columns.remove('City_Name')
    columns.remove('Date')
    columns.remove('rolling_average_7_days')
    columns.remove('today_verified_cases')
    for m in range(5, 15):
        print(f'm={m}')
        experiment_m_features(m, examples, columns)


def experiment_m_features(m, examples, columns):
    """
    This function randomize m features and test the knn based on them and print it's results
    :param m: number of features to test
    :param examples: dataset to split to train and test
    :param columns: columns to choose from
    :return: None
    """
    examples['Date'] = pd.to_datetime(examples['Date'])
    for i in range(30):
        test_columns = sample(columns, m) + [params.Y]

        split_date = datetime(2021, 7, 23)
        train_examples = examples[examples['Date'] < split_date][test_columns]
        test_examples = examples[examples['Date'] >= split_date][test_columns]
        knn_output = run_knn(K, train_examples, test_examples)
        print(f'columns={test_columns}')
        print_results(knn_output)


def experiment_k_kfold(examples):
    """
    This is an old function that we used to do regular k-fold before we realized that the k-fold can use the future to
    predict the past.
    :param examples: examples for train and test
    :return: None
    """
    kf = KFold(n_splits=5)
    K = range(3, 25)
    accuracies = []
    for k in K:
        print(f'k:{k}')
        sum = 0
        for train_index, test_index in kf.split(examples):
            train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]

            acc = run_knn(k, train_examples, test_examples)
            sum += acc['acc']
        avg_accuracy = sum / kf.n_splits
        accuracies.append(avg_accuracy)
        print(f'K:{k}, average accuracy: {avg_accuracy}')

    plt.plot(K, accuracies)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.show()


def experiment_k(train, test):
    """
    This function is used to test k in specific range. that we plot the results on a graph
    :param train: train data set
    :param test: test data set
    :return: None
    """
    train = train[best_columns]
    test = test[best_columns]
    K = range(3, 100)
    r2_accs = []
    mae_accs = []
    for k in K:
        knn_output = run_knn(k, train, test)
        r2_accs.append(knn_output['acc'])
        mae_accs.append(knn_output['mae'])

    plt.plot(K, r2_accs)
    plt.xlabel('K')
    plt.ylabel('r2')
    plt.title('K Accuracy')
    plt.show()

    plt.plot(K, mae_accs)
    plt.xlabel('K')
    plt.ylabel('mae')
    plt.show()


def experiment_weights(train, test):
    """
    This function test the 2 different weight function of knn algorithm
    :param train: train data set for knn
    :param test: test data set for knn
    :return: None
    """
    print('testing best weights')
    print('uniform:')
    run_knn_best_columns(K, train, test, 'uniform')
    print('distance:')
    run_knn_best_columns(K, train, test, 'distance')


def run_knn_on_small_cities(population, k, train_df, test_df):
    """
    This function filter the test set and testing only on small cities (cities that their population is smaller than
    population parameter
    :param population: max population of the records in the test set
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    population_df = pd.read_csv('../../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] <= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    print_results(knn_output, f'knn_on_small_cities population < {population}')
    diff = pd.DataFrame(abs(knn_output['y_test'] - knn_output['y_predicted']))
    plot_diff_graph(diff, '#sample', 'new cases diff', 'Small Cities New Cases Error', knn_output['test_mean'])


def run_knn_on_big_cities(population, k, train_df, test_df):
    """
    This function filter the test set and testing only on big cities (cities that their population is bigger than
    population parameter
    :param population: max population of the records in the test set
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    population_df = pd.read_csv('../../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] >= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    print_results(knn_output, f'knn_on_big_cities population > {population}')
    diff = pd.DataFrame(abs(knn_output['y_test'] - knn_output['y_predicted']))
    plot_diff_graph(diff, '#sample', 'new cases diff', 'Big Cities New Cases Error', knn_output['test_mean'])


def run_knn_on_small_new_cases(new_cases, k, train_df, test_df):
    """
    This function tests our output algorithm only on test set where y_test is smaller then new_cases param
    :param new_cases: max new cases in y_true of test_df
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    test_df = test_df[test_df[params.Y] <= new_cases]
    knn_output = run_knn(k, train_df, test_df)
    print_results(knn_output, f'knn_on_small_new_cases new_cases < {new_cases}')
    diff = pd.DataFrame(abs(knn_output['y_test'] - knn_output['y_predicted']))
    plot_diff_graph(diff, '#sample', 'new cases diff', 'Small New Cases Diff', knn_output['test_mean'])


def run_knn_on_big_new_cases(new_cases, k, train_df, test_df):
    """
    This function tests our output algorithm only on test set where y_test is bigger then new_cases param
    :param new_cases: min new cases in y_true of test_df
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    test_df = test_df[test_df[params.Y] >= new_cases]

    knn_output = run_knn(k, train_df, test_df)
    print_results(knn_output, f'knn_on_big_new_cases new_cases > {new_cases}')
    diff = pd.DataFrame(abs(knn_output['y_test'] - knn_output['y_predicted']))
    plot_diff_graph(diff, '#sample', 'new cases diff', 'Big Cities New Cases Diff', knn_output['test_mean'])


def run_knn_on_colour(colour, k, train_df, test_df):
    """
    This function test our output algorithm on the input 'Ramzor' colour
    :param colour: which colour we want to test
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    test_df = test_df[test_df['colour'] == colour]

    knn_output = run_knn(k, train_df, test_df)
    if colour == 0:
        colour_name = 'green'
    elif colour == (1/3):
        colour_name = 'yellow'
    elif colour == (2/3):
        colour_name = 'orange'
    elif colour == 1:
        colour_name = 'red'

    print_results(knn_output, f'knn_on_colour colour={colour_name}')
    diff = pd.DataFrame(abs(knn_output['y_test'] - knn_output['y_predicted']))
    plot_diff_graph(diff, '#sample', 'new cases diff', f'{colour_name} Cities New Cases Diff', knn_output['test_mean'])


def run_knn_on_dates(k, train_df, test_df, start_date, end_date, best_columns):
    """
    This function filter the test set for specific date range.
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :param start_date: min date for the test set
    :param end_date: max date for the test set
    :param best_columns: knn columns
    :return: None
    """
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]
    test_df = test_df[best_columns]
    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    my_acc = knn_output['my_acc']
    print(f'knn_on_big_new_cases accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, my_acc={my_acc}, start_date {start_date}, , end_date {end_date}')


def experiment_subset_data(k, train_df, test_df):
    """
    This function print stats and graphs of the different fields we tested the performance of our output algorithm
    :param k: k value of knn
    :param train_df: train for knn
    :param test_df: test for knn
    :return: None
    """
    train_df = train_df[best_columns]
    test_df = test_df[best_columns]

    run_knn_on_small_cities(30000, k, train_df, test_df)
    run_knn_on_big_cities(30000, k, train_df, test_df)
    run_knn_on_small_new_cases(50, k, train_df, test_df)
    run_knn_on_big_new_cases(50, k, train_df, test_df)
    run_knn_on_colour(0, k, train_df, test_df)
    run_knn_on_colour((1/3), k, train_df, test_df)
    run_knn_on_colour((2/3), k, train_df, test_df)
    run_knn_on_colour(1, k, train_df, test_df)


def investigate_corona_df(corona_df, train_df, test_df):
    """
    This function show stats and graphs of our data
    :param corona_df: the entire data
    :param train_df: train data set
    :param test_df: test data set
    :return: None
    """
    train_min_date = train_df['Date'].min().strftime('%d-%m-%Y')
    train_max_date = train_df['Date'].max().strftime('%d-%m-%Y')

    test_min_date = test_df['Date'].min().strftime('%d-%m-%Y')
    test_max_date = test_df['Date'].max().strftime('%d-%m-%Y')

    print(f'Train date range: {train_min_date} - {train_max_date}')
    print(f'Train date range: {test_min_date} - {test_max_date}')

    plt.figure(figsize=(10, 4))
    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['today_verified_cases'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['today_verified_cases'].values, label="BNEI BRAK")
    plt.plot(corona_df[corona_df['City_Code'] == 3730]['Date'].values, corona_df[corona_df['City_Code'] == 3730]['today_verified_cases'].values, label="Givat Zeev")
    plt.plot(corona_df[corona_df['City_Code'] == 53]['Date'].values, corona_df[corona_df['City_Code'] == 53]['today_verified_cases'].values, label="Atlit")
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('New Cases')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['rolling_average_7_days'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['rolling_average_7_days'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['rolling_average_7_days'].values, label="BNEI BRAK")
    plt.plot(corona_df[corona_df['City_Code'] == 3730]['Date'].values, corona_df[corona_df['City_Code'] == 3730]['rolling_average_7_days'].values, label="Givat Zeev")
    plt.plot(corona_df[corona_df['City_Code'] == 53]['Date'].values, corona_df[corona_df['City_Code'] == 53]['rolling_average_7_days'].values, label="Atlit")
    plt.xlabel('Date')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('Daily new cases Rolling Avg (7 Days)')
    plt.show()

    run_knn_best_columns(K, train_df, test_df)

    train_df['today_verified_cases'] = train_df['rolling_average_7_days']
    test_df['today_verified_cases'] = test_df['rolling_average_7_days']
    print('rolling average 7 days:')
    run_knn_best_columns(K, train_df, test_df)


def run_knn_best_columns(k, train, test, weights='uniform'):
    """
    This function run knn with the best features found in the experiments
    :param k: k value of knn
    :param train: train set for knn
    :param test: test set for knn
    :param weights: weights function for knn
    :return: None
    """
    train_df = train[best_columns]
    test_df = test[best_columns]
    knn_output = run_knn(k, train_df, test_df, weights)
    print_results(knn_output, 'knn after learning')


def run_knn_all_columns(k, train, test, weights='uniform'):
    """
    This function run and print the results of raw knn (with all features)
    :param k: k value of knn
    :param train: train data set for knn
    :param test: test data set for knn
    :param weights: weights function for knn
    :return: None
    """
    cols.remove('City_Name')
    cols.remove('Date')
    cols.remove('rolling_average_7_days')
    # cols.remove('today_verified_cases')
    train_df = train[cols]
    test_df = test[cols]
    knn_output = run_knn(k, train_df, test_df, weights)
    print_results(knn_output, 'knn all columns')


def print_results(knn_output, string_msg=None):
    """
    This function prints the results based on knn_output
    :param knn_output: the results of run_knn function
    :param string_msg: the msg to show in the prefix of the message
    :return: None
    """
    acc = round(knn_output['acc'], 3)
    mae = round(knn_output['mae'], 3)
    test_mean = round(knn_output['test_mean'], 3)
    test_len = len(knn_output['y_test'])
    if string_msg is None:
        print(f'knn accuracy: R2:{acc}, mae: {mae}, y_test_mean={test_mean}, test_len={test_len}')
    else:
        print(f'{string_msg}: R2:{acc}, mae: {mae}, y_test_mean={test_mean}, test_len={test_len}')


def evaluate_all_columns(k, train_df):
    """
    This function run all columns only on train data set (splits it internally to train and test)
    :param k: k value of knn
    :param train_df: data set to train and test the knn
    :return: None
    """
    # Leave Y column on
    cols.remove('City_Name')
    cols.remove('Date')
    cols.remove('rolling_average_7_days')
    # cols.remove('today_verified_cases')
    split_date = datetime(2021, 8, 23)
    train_examples = train_df[train_df['Date'] < split_date][cols]
    test_examples = train_df[train_df['Date'] >= split_date][cols]
    knn_output = run_knn(k, train_examples, test_examples)
    print_results(knn_output)


def print_data_stats(train_df, test_df):
    """
    Prints date ranges and mean of the data
    :param train_df: train data set
    :param test_df: test data set
    :return:
    """
    train_min_date = train_df['Date'].min().strftime('%d-%m-%Y')
    train_max_date = train_df['Date'].max().strftime('%d-%m-%Y')

    test_min_date = test_df['Date'].min().strftime('%d-%m-%Y')
    test_max_date = test_df['Date'].max().strftime('%d-%m-%Y')

    test_mean = round(test_df[params.Y].mean(), 3)

    print(f'Train date range: {train_min_date} - {train_max_date}')
    print(f'Train date range: {test_min_date} - {test_max_date}')
    print(f'test mean: {test_mean}')


if __name__ == "__main__":
    """
    Here you can run all the above code.
    There are 2 parts, the uncommented code which is important for each run.
    And the commented code between the '####' lines.
    
    In each run you need to uncomment 1 block between 2 following '####' and run it.
    Each block is stand alone and will run normally alone.   
    """
    # Load data
    corona_df = pd.read_csv('../../Preprocess/corona_df.csv')
    train_df = pd.read_csv('../../Preprocess/train_df.csv')
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    corona_df.sort_values(by=['Date'], inplace=True)
    train_df.sort_values(by=['Date'], inplace=True)
    test_df.sort_values(by=['Date'], inplace=True)

    print_data_stats(train_df, test_df)
    ################
    # investigate_corona_df(corona_df, train_df, test_df)
    # exit(0)
    ################
    # evaluate_all_columns(K, train_df) # run only on train
    # exit(0)
    ################
    # run_knn_all_columns(10, train_df, test_df)
    # exit(0)
    ################
    # experiment_features(train_df, list(train_df.columns))
    # exit(0)
    ################
    run_knn_best_columns(40, train_df, test_df, 'distance')
    exit(0)
    ################
    # experiment_k(train_df, test_df)
    # exit(0)
    ################
    # experiment_rolling_train(K, train_df)
    # exit(0)
    ################
    # experiment_weights(train_df, test_df)
    # exit(0)
    ################
    # corona_df = corona_df[best_columns]
    # run_knn_with_split(K, corona_df)
    ################
    # plot_graphs(K, train_df, test_df)
    ################
    # experiment_subset_data(K, train_df, test_df)
