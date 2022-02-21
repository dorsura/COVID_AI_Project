import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from Algorithms.DT import DTAuxiliaries
from Algorithms.DT import DTPrinting
from Preprocess import utils

'''
This part was created in order to select the most relevant features for Decision Tree Regressor and Random Forest Regressor models.
If you wish to run this section with pruning, initialize 'pruning_option' parameter with 'With Pruning'. Otherwise,
use 'Without Pruning'.
'''

#######################################################################################################################
#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################


def get_best_RFE_results_for_DTR_or_RFR(regressor_type, leaf_samples_range, avg_features_flag):
    best_score = 0
    best_index = 1
    best_features = []
    regressor = 'DTR_or_RFR'
    sel = 'RFE'
    DT_results_vec = []
    DT_scores_vec = []
    RF_results_vec = []
    RF_scores_vec = []

    params = utils.DataParams()
    if avg_features_flag == "Without Avg Features":
        corona_df = DTAuxiliaries.get_corona_df()
    if avg_features_flag == "With Avg Features":
        corona_df = DTAuxiliaries.get_corona_df_with_avg_cumulated_verified_cases()

    train_df = corona_df[corona_df['Date'] < params.split_date_for_dt]
    train_df = train_df.drop(['City_Name', 'Date'], axis=1)
    train_df_no_pred_col = train_df.drop(['today_verified_cases'], axis=1)

    for leaf_samples in range(1, leaf_samples_range+1):
        for index in range(2, len(train_df_no_pred_col.columns)):

            train_df, test_df = DTAuxiliaries.get_train_and_test_df(corona_df, params)
            X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

            if regressor_type == "DecisionTreeRegressor":
                sel = RFE(DecisionTreeRegressor(random_state=1, min_samples_leaf=leaf_samples), n_features_to_select=index)
                sel.fit(X_train, Y_train)
                regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=leaf_samples)
            if regressor_type == "RandomForestRegressor":
                sel = RFE(RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=leaf_samples), n_features_to_select=index)
                sel.fit(X_train, Y_train)
                regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=leaf_samples)

            X_train_rfe = sel.transform(X_train)
            regressor.fit(X_train_rfe, Y_train)
            selected_features = [col for col in train_df_no_pred_col]
            features = np.array(selected_features)[sel.get_support()]

            train_df = train_df[[col for col in train_df.columns if col in features or col == 'today_verified_cases']]
            test_df = test_df[[col for col in test_df.columns if col in features or col == 'today_verified_cases']]

            X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

            score = regressor.score(X_test, Y_test)

            if regressor_type == "DecisionTreeRegressor" and leaf_samples == 1:
                DT_scores_vec.append(score)
            if regressor_type == "RandomForestRegressor" and leaf_samples == 1:
                RF_scores_vec.append(score)

            if best_score < score:
                best_index = index
                best_score = score
                best_features = features

        if regressor_type == "DecisionTreeRegressor":
            DT_results_vec.append((best_score, best_index, best_features, leaf_samples))
        if regressor_type == "RandomForestRegressor":
            RF_results_vec.append((best_score, best_index, best_features, leaf_samples))

        best_index = 0
        best_score = 0
        best_features = []

    if regressor_type == "DecisionTreeRegressor":
        if leaf_samples_range == 1:
            DTPrinting.print_RFE_scores(DT_scores_vec, "DecisionTreeRegressor")
        return max(DT_results_vec)
    if regressor_type == "RandomForestRegressor":
        if leaf_samples_range == 1:
            DTPrinting.print_RFE_scores(RF_scores_vec, "RandomForestRegressor")
        return max(RF_results_vec)


def get_RFE_best_features_and_best_min_samples_leaf(pruning_flag, avg_features_flag):
    DT_best_results = []
    RF_best_results = []

    if pruning_flag == "Without Pruning":
        DT_best_results = get_best_RFE_results_for_DTR_or_RFR("DecisionTreeRegressor", 1, avg_features_flag)

        DTPrinting.print_best_RFE_results(DT_best_results, "DecisionTreeRegressor", pruning_flag)

        RF_best_results = get_best_RFE_results_for_DTR_or_RFR("RandomForestRegressor", 1, avg_features_flag)
        DTPrinting.print_best_RFE_results(RF_best_results, "RandomForestRegressor", pruning_flag)

    if pruning_flag == "With Pruning":
        DT_best_results = get_best_RFE_results_for_DTR_or_RFR("DecisionTreeRegressor", 10, avg_features_flag)
        DTPrinting.print_best_RFE_results(DT_best_results, "DecisionTreeRegressor", pruning_flag)

        RF_best_results = get_best_RFE_results_for_DTR_or_RFR("RandomForestRegressor", 10, avg_features_flag)
        DTPrinting.print_best_RFE_results(RF_best_results, "RandomForestRegressor", pruning_flag)

    return DT_best_results[2], DT_best_results[3], RF_best_results[2], RF_best_results[3]


def run_FeaturesSelection(pruning_flag, avg_features_flag, only_features_importances_graph_flag):

    if only_features_importances_graph_flag == True:
        DTPrinting.print_features_importances("DecisionTreeRegressor")
        DTPrinting.print_features_importances("RandomForestRegressor")
        return

    DT_best_features, DT_best_min_samples_leaf, RF_best_features, RF_best_min_samples_leaf = get_RFE_best_features_and_best_min_samples_leaf(pruning_flag, avg_features_flag)

    params = utils.DataParams()
    if avg_features_flag == "Without Avg Features":
        corona_df = DTAuxiliaries.get_corona_df()
    if avg_features_flag == "With Avg Features":
        corona_df = DTAuxiliaries.get_corona_df_with_avg_cumulated_verified_cases()

    train_df, test_df = DTAuxiliaries.get_train_and_test_df(corona_df, params)
    train_df = train_df[[col for col in train_df.columns if col in DT_best_features or col == 'today_verified_cases']]
    test_df = test_df[[col for col in test_df.columns if col in DT_best_features or col == 'today_verified_cases']]

    X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=DT_best_min_samples_leaf)
    DT_regressor.fit(X_train, Y_train)
    DT_test_set_res = DT_regressor.score(X_test, Y_test)

    print("PART FeaturesSelection Final Results: ")
    print(f"DecisionTreeRegressor Test Set Score : {round(DT_test_set_res, 3)}")

    train_df, test_df = DTAuxiliaries.get_train_and_test_df(corona_df, params)
    train_df = train_df[[col for col in train_df.columns if col in RF_best_features or col == 'today_verified_cases']]
    test_df = test_df[[col for col in test_df.columns if col in RF_best_features or col == 'today_verified_cases']]

    X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
    RF_regressor.fit(X_train, Y_train)
    RF_test_set_res = RF_regressor.score(X_test, Y_test)

    print(f"RandomForestRegressor Test Set Score : {round(RF_test_set_res, 3)}")


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################

