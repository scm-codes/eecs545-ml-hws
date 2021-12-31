import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt

# You have have to install the libraries below.
# sklearn, csv
import csv

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# The csv file air-quality-train.csv contains the training data.
# After loaded, each row of X_train will correspond to CO, NO2, O3, SO2.
# The vector y_train will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_train = []
y_train = []

with open('air-quality-train.csv', 'r') as air_quality_train:
    air_quality_train_reader = csv.reader(air_quality_train)
    next(air_quality_train_reader)
    for row in air_quality_train_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_train.append([row[1], row[2], row[3], row[4]])
        y_train.append(row[5])
        
# The csv file air-quality-test.csv contains the testing data.
# After loaded, each row of X_test will correspond to CO, NO2, O3, SO2.
# The vector y_test will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_test = []
y_test = []

with open('air-quality-test.csv', 'r') as air_quality_test:
    air_quality_test_reader = csv.reader(air_quality_test)
    next(air_quality_test_reader)
    for row in air_quality_test_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_test.append([row[1], row[2], row[3], row[4]])
        y_test.append(row[5])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# TODOs for part (a)
#    1. Use SVR loaded to train a SVR model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset



# TODOs for part (b)
#    1. Use KernelRidge to train a Kernel Ridge  model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset 


# Use this seed.
seed = 0
np.random.seed(seed) 

K = 5 #The number of folds we will create 

# TODOs for part (c)
#   1. Create a partition of training data into K=5 folds 
#   Hint: it suffice to create 5 subarrays of indices   


# Specify the grid search space 
reg_range = np.logspace(-1,1,3)     # Regularization paramters
kpara_range = np.logspace(-2, 0, 3) # Kernel parameters 

# TODOs for part (d)
#    1.  Select the best parameters for both SVR and KernelRidge based on k-fold cross-validation error estimate (use RMSE as the performance metric)
#    2.  Print the best paramters for both SVR and KernelRidge selected  
#    3.  Train both SVR and KernelRidge on the full training data with selected best parameters 
#    4.  Print both the RMSE on the test dataset of SVR and KernelRidge 