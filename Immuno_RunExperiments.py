# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:34:24 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Preprocess Immunotherapy dataset 

"""

from urllib.request import urlretrieve

# Import pandas
import pandas as pd
# Import numpy
import numpy as np
import os 
from Models.kfold_CV_try import train_test_split 
from Models.kfold_CV_try import kfold_cross_validation
from Models.kfold_CV_try import train_validation_split
import sklearn.preprocessing as preprocessing          # used for one hot encoding 
import seaborn as sns
import matplotlib.pyplot as plt

dataFolderPath = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Dataset_Folder\\Dataset4_Immunotherapy'
os.chdir (dataFolderPath)

df_immuno_all = pd.read_csv('Immunotherapy.csv', sep=',') # adult data training set (including the validation set)

url_immuno='https://archive.ics.uci.edu/ml/machine-learning-databases/00428/Immunotherapy.xlsx'
result = df_immuno_all.copy()
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

temp_im = df_immuno_all.values
R_im,C_im = temp_im.shape
print(R_im);print(C_im)

# Split array into design matrix and labels
im_labels = temp_im[:, C_im-1]
print(im_labels)

# Remove labels to get design matrix
immuno_design_matrix= np.delete(temp_im, C_im-1, 1)
print(immuno_design_matrix)

###############################################################################
# 5-FOLD CROSS VALIDATION DATA SPLITTING 

immuno_all = df_immuno_all.to_numpy()
immuno_all = df_immuno_all.values
# labels 
immuno_labels = immuno_all[:,7]
# design matrix
immuno_desMat = immuno_all[:,0:7]

# splittting data
xTrain_im, xTest_im, yTrain_im, yTest_im = train_test_split(immuno_desMat, immuno_labels, test_size=0.2, random_state = 0)

# k-fold cross validation 
folds = 5
#dataset_split_im, cv_train_data_im,cv_train_label_im= kfold_cross_validation(dataset_im,folds)
dataset_split_im, cv_train_data_im,cv_train_label_im= kfold_cross_validation(xTrain_im,yTrain_im,folds)

    
# cv_train_data has k (5) folds including the 4 folds for training set and 1 fold for testing set 
# cv_train_label has k (5) folds for the corresponding labels for the 5 folds in cv_train_data
  
# training the model k times so change the index of training_data and training_labels 
# training data: concatenate k-1 folds, validation data for 1 fold: evaluate for k times, uncomment the rest of the lines when testing 
validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,1)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,2)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,3)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,4)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,5)

##############################################################################
# MODEL IMPLEMENTATION (sklearn)
##############################################################################

##############################################################################
## LOGISTIC REGRESSION MODEL IMPLEMENTATION AND TESTING 
import Models.logisticRegression as logReg
immuno_log=logReg.Logistic_Regression(0.5,"Immunotherapy","binary")
print(immuno_log.name)
log_immu_results = immuno_log.fit(immuno_design_matrix, im_labels, learning_rate = 0.5, term = 1)

imlr = lr.Logistic_Regression(0.02,"Immunotherapy","binary")
im_params = lr.fit(imlr,immuno_design_matrix, im_labels, 0.01, 1e-2)
print(im_params)

predictions_im = lr.predict(imlr,im_params,immuno_design_matrix)
print(predictions_im)

