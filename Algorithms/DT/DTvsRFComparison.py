from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from Algorithms.DT import DTAuxiliaries
from Algorithms.DT import DTPrinting
from Preprocess import utils

'''
This part was created in order to make a Decision Tree Regressor vs. Random Forest Regressor comparison.
If you wish to run this section with 'City_Code' feature, initialize 'city_code_flag' with 'With City_Code'. Otherwise,
use 'Without City_Code'.
'''

#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


def run_DTvsRFComparison(city_code_flag, city_code_comparison_flag):
    params = utils.DataParams()
    corona_df = DTAuxiliaries.get_corona_df()
    if city_code_flag == "With City_Code":
        train_df, test_df = DTAuxiliaries.get_train_and_test_df_part_A(corona_df, params, "With City_Code")
    if city_code_flag == "Without City_Code":
        train_df, test_df = DTAuxiliaries.get_train_and_test_df_part_A(corona_df, params, "Without City_Code")

    X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1)
    DT_regressor.fit(X_train, Y_train)
    DT_test_set_res = DT_regressor.score(X_test, Y_test)
    DT_Y_pred = DT_regressor.predict(X_test)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1)
    RF_regressor.fit(X_train, Y_train)
    RF_test_set_res = RF_regressor.score(X_test, Y_test)
    RF_Y_pred = RF_regressor.predict(X_test)

    print("PART DTvsRFComparison Final Results: ")

    if city_code_comparison_flag == True:
        if city_code_flag == "With City_Code":
            print("With City_Code Feature: \n")
        if city_code_flag == "Without City_Code":
            print("Without City_Code Feature: \n")

    print(f"DecisionTreeRegressor Test Set Score : {round(DT_test_set_res, 3)}")
    print(f"RandomForestRegressor Test Set Score : {round(RF_test_set_res, 3)}")

    if city_code_flag == "With City_Code" and city_code_comparison_flag == False:
        DTPrinting.print_y_test_vs_y_predict(Y_test, DT_Y_pred, RF_Y_pred)


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################

