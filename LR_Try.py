# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:37:59 2020

@author: Admin
"""


# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

# Assign url of file: url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'

# Save file locally
urlretrieve(url, 'ionosphere-data.csv')

# Read file into a DataFrame and print its head
df = pd.read_csv('ionosphere-data.csv', sep=',')
print(df)

#df.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14","15","16","17","18","19","20","21", "22", "23","24","25","26","27","28","29","30","31","32","33","34","Labels"]
result = df.copy()
encoders = {}
for column in result.columns:
    if result.dtypes[column] == np.object:
        encoders[column] = preprocessing.LabelEncoder()
        result[column] = encoders[column].fit_transform(result[column])

# Calculate the correlation and plot it
# encoded_data, _ = number_encode_features(df)
    # show heatmap - exploring correlations 
encoded_data = result 
sns.heatmap(encoded_data.corr(), square=True)
plt.show()
encoded_data.tail(5)


# Convert to numpy
#df.to_numpy()
temp = df.values
R,C = temp.shape
print(C)

# Split array into design matrix and labels
ionosphere_labels = temp[:, C-1]
print(ionosphere_labels)

# Remove labels to get design matrix
ionosphere_design_matrix= np.delete(temp, C-1, 1)
print(ionosphere_design_matrix)

# LR without kfold cross validation 
import Models.logisticRegression as lr
ionslr = lr.Logistic_Regression(0.02,"Ionosphere","binary")
params = lr.fit(ionslr,ionosphere_design_matrix, ionosphere_labels, 0.01, 1e-2)
print(params)
predictions = lr.predict(ionslr,params,ionosphere_design_matrix)
print(predictions)


# LR with kfold cross validation 
from Models.kfold_CV_try import train_test_split 
from Models.kfold_CV_try import kfold_cross_validation
from Models.kfold_CV_try import train_validation_split
# split the whole dataset into training and testing sets 
xTrain_ion, xTest_ion, yTrain_ion, yTest_ion = train_test_split(ionosphere_design_matrix, ionosphere_labels, test_size=0.2, random_state = 0)

# k-fold cross validation 
folds = 5 # delete folds later when embedded in function input 
dataset_split_in, cv_train_data_ion,cv_train_label_ion= kfold_cross_validation(xTrain_ion,yTrain_ion,folds)

validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = train_validation_split(cv_train_data_ion,cv_train_label_ion,1)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,2)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,3)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,4)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,5)


