import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from Algorithms.DT import DTAuxiliaries
from Preprocess import utils
from sklearn import metrics


font = {'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)


#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


def print_y_test_vs_y_predict(Y_test, DT_regressor_results, RF_regressor_results):
    x_ax = range(len(Y_test[10:50]))
    plt.plot(x_ax, Y_test[10:50], linewidth=1, label="Original")
    plt.plot(x_ax, DT_regressor_results[10:50], linewidth=1.1, label="DecisionTreeRegressor Predicted")
    plt.plot(x_ax, RF_regressor_results[10:50], linewidth=1.1, label="RandomForestRegressor Predicted")
    plt.title("DecisionTreeRegressor and RandomForestRegressor - Y_test vs. Y_predicted")
    plt.xlabel('Number of Sample')
    plt.ylabel('Value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################


def print_RFE_scores(regressor_scores, regressor_type):
    num_of_features_range = []
    for i in range(1, len(regressor_scores)+1):
        num_of_features_range.append(i)
    plt.plot(num_of_features_range, regressor_scores, color='b', linewidth=3, marker='8')
    plt.xlabel('Number of features selected', fontsize=15)
    plt.ylabel('% Score', fontsize=15)

    if regressor_type == "DecisionTreeRegressor":
        plt.title('DecisionTreeRegressor - Recursive Feature Elimination')
    if regressor_type == "RandomForestRegressor":
        plt.title('RandomForestRegressor - Recursive Feature Elimination')

    plt.show()


def print_best_RFE_results(regressor_best_features, regressor_type, pruning_option):
    if regressor_type == "DecisionTreeRegressor":
        print('Regressor Type: DecisionTreeRegressor')
    if regressor_type == "RandomForestRegressor":
        print('Regressor Type: RandomForestRegressor')
    print('Best Score: ', round(regressor_best_features[0], 3))
    print('Best Number Of Features: ', regressor_best_features[1])
    print("Features Were Selected: ", regressor_best_features[2])
    if pruning_option == "With Pruning":
        print("Min Samples in Leaf: ", regressor_best_features[3])
    print('\n')


def print_features_importances(regressor_type):
    params = utils.DataParams()
    corona_df = DTAuxiliaries.get_corona_df()
    train_df, test_df = DTAuxiliaries.get_train_and_test_df(corona_df, params)
    X_train, Y_train, X_test, Y_test = DTAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    regressor = 'DTR_or_RFR'
    if regressor_type == "DecisionTreeRegressor":
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('DecisionTreeRegressor - Features importances')
    if regressor_type == "RandomForestRegressor":
        regressor = RandomForestRegressor(n_estimators=10, random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('RandomForestRegressor - Features importances')

    features = [col for col in train_df if col != 'today_verified_cases']
    f_i = list(zip(features, regressor.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    plt.show()


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################


def print_result_metrics(regressor_type, y_true, y_pred):
    if regressor_type == "DecisionTreeRegressor":
        print('###----Metrics for DecisionTreeRegressor accuracy---###')
    if regressor_type == "RandomForestRegressor":
        print('###----Metrics for RandomForestRegressor accuracy---###')
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    med = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('Test Data Mean: ', round(np.mean(y_true), 3))
    print('Mean Absolute Error: ', round(mae, 3))
    print('Mean Squared Error: ', round(mse, 3))
    print('Mean Absolute Percentage Error: ', round(mape, 3))
    print('Root Mean Squared Error: ', round(np.sqrt(mse), 3))
    print('Median Absolute Error: ', round(med, 3))
    print('R2 score: ', round(r2, 3))


#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################
#######################################################################################################################

