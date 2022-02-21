## Installation

### Requirements 
* Python version: 3.7

### Installation Guide

```bash
$ git clone https://github.com/Eyallotan/COVID_AI_Project.git
$ cd COVID_AI_Project
$ set up a virtual env
$ run "pip install -r txt.libs" 
$ cd Algorithms
$ Run the relevant code
```

## Usage

### Algorithms  
This folder contains all of the different algorithms and classifiers we have implemented. 
1. Decision tree (DT) - This folder contains the basic DT classifiers and code that runs the feature selection
algoritm, pruning and other experiments. You can run all experiments from ```DT.py```.
2. KNN - This folder contains the KNN classifer, as well as code for running all of the KNN experminets (feature 
selection, parameter tuning etc.). You can run all experiments from ```knn.py```.
3. time_series - This folder contains all of the time series algorithms:
* ```DataTransformation.py``` - This file contains the implementation of the time series transformations we offer in order to transform your time series to stationary. Each transformation has an inverse transformation that can be applied to it. This library also supports applying several transformations one after the other. 
* ```AlgoRunner.py``` - Infrastructure and execution engine for running various time series forecasting algorithms. This runner supports all SARIMA family models (MA/AR/ARMA/ARIMA/SARIMA), and also implements the rolling forecast algorithm (for model retraining). 
* ```TimeSeriesAnalysis.py``` - This includes the main function that runs all of the experiments and demonstrations for the time sereis analysis chapter. In addition here is where we preprocess all of the data needed for the time series algorithms.
	   
### Resources 
This folder contains all of the raw data files we have imported. These data files are used by the preprocessing code
in order to generate our datasets and add custom features.

### Preprocess  
This folder contains the code related to the preprocessing phase. There is no need to run this code since 
the dataset we are using (corona_df.csv) was already created and added to the repository. 
In order to re-run the preprocessing phase:
1. Run ```utils.py```. This will handle the first preprocess phase of the raw data.
2. Run ```create_main_df.py```. This will create all of our custom features and add them to the dataset, and eventually
create the corona_df.csv. 

## License

All of the code in this project has been written by Eyal Lotan, Dor Sura and Tomer Haber. All external data sources are listed in the [project report](https://github.com/Eyallotan/COVID_AI_Project/blob/master/Project%20report.pdf).
