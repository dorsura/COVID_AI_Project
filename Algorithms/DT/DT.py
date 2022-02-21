from Algorithms.DT import DTvsRFComparison
from Algorithms.DT import FeaturesSelection
from Algorithms.DT import BestModel

if __name__ == "__main__":
    ################################################### PART A ############################################################
    DTvsRFComparison.run_DTvsRFComparison(city_code_flag="With City_Code", city_code_comparison_flag=False)
    print("====================================================================================")

    '''
    Sections B and C may take several hours to run and print their results 
    '''
    ################################################### PART B ############################################################
    FeaturesSelection.run_FeaturesSelection(pruning_flag="Without Pruning", avg_features_flag="Without Avg Features",
                                            only_features_importances_graph_flag=True)
    DTvsRFComparison.run_DTvsRFComparison(city_code_flag="With City_Code", city_code_comparison_flag=True)
    DTvsRFComparison.run_DTvsRFComparison(city_code_flag="Without City_Code", city_code_comparison_flag=True)
    print("====================================================================================")
    FeaturesSelection.run_FeaturesSelection(pruning_flag="Without Pruning", avg_features_flag="Without Avg Features",
                                            only_features_importances_graph_flag=False)
    print("====================================================================================")
    FeaturesSelection.run_FeaturesSelection(pruning_flag="With Pruning", avg_features_flag="Without Avg Features",
                                            only_features_importances_graph_flag=False)
    print("====================================================================================")

    ################################################### PART C ############################################################
    FeaturesSelection.run_FeaturesSelection(pruning_flag="With Pruning", avg_features_flag="With Avg Features",
                                            only_features_importances_graph_flag=False)
    print("====================================================================================")

    ################################################### SUMMARY ###########################################################
    BestModel.run_BestModel()
