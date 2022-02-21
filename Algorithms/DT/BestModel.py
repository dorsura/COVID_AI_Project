from Algorithms.DT import DTPrinting
from Algorithms.DT import DTAuxiliaries
from sklearn.ensemble import RandomForestRegressor
from Preprocess import utils


#######################################################################################################################
#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################


def run_BestModel():
    RF_best_min_samples_leaf = 8

    RF_best_features = ['City_Code', 'Cumulative_verified_cases', 'Cumulated_deaths', 'colour',
                        'final_score', 'vaccinated_dose_1_total', 'verified_cases_2_days_ago_avg',
                        'verified_cases_8_days_ago_avg']

    params = utils.DataParams()
    corona_df = DTAuxiliaries.get_corona_df_with_avg_cumulated_verified_cases()

    train_df, test_df = DTAuxiliaries.get_train_and_test_df(corona_df, params)
    train_df = train_df[[col for col in train_df.columns if col in RF_best_features or col == 'today_verified_cases']]
    test_df = test_df[[col for col in test_df.columns if col in RF_best_features or col == 'today_verified_cases']]

    X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
    RF_regressor.fit(X_train, Y_train)
    RF_Y_pred = RF_regressor.predict(X_test)
    DTPrinting.print_result_metrics("RandomForestRegressor", Y_test, RF_Y_pred)


#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################
#######################################################################################################################

