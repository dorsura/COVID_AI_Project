import pandas as pd


def get_corona_df():
    corona_df = pd.read_csv('corona_df_parts_A_B.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    corona_df = corona_df.sort_values(ascending=False, by=['Date'])

    return corona_df

def get_corona_df_with_avg_cumulated_verified_cases():
    corona_df = pd.read_csv('corona_df_part_C.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    corona_df = corona_df.sort_values(ascending=False, by=['Date'])

    return corona_df

def get_train_and_test_df(df, params):
    train_df = df[df['Date'] < params.split_date_for_dt]
    test_df = df[df['Date'] >= params.split_date_for_dt]
    train_df = train_df.drop(['City_Name', 'Date'], axis=1)
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    return train_df, test_df

def get_train_and_test_df_part_A(df, params, city_code_flag):
    train_df = df[df['Date'] < params.split_date_for_dt]
    test_df = df[df['Date'] >= params.split_date_for_dt]
    if city_code_flag == "With City_Code":
        train_df = train_df.drop(['City_Name', 'Date'], axis=1)
        test_df = test_df.drop(['City_Name', 'Date'], axis=1)
    if city_code_flag == "Without City_Code":
        train_df = train_df.drop(['City_Name', 'Date', 'City_Code'], axis=1)
        test_df = test_df.drop(['City_Name', 'Date', 'City_Code'], axis=1)

    return train_df, test_df

def get_X_and_Y_tarin_test_sets(train_df, test_df):
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    return X_train, Y_train, X_test, Y_test

