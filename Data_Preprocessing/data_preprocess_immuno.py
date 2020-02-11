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
from sklearn.model_selection import train_test_split
from kfold_CV_try import kfold_cross_validation

dataFolderPath = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Dataset_Folder\\Dataset4_Immunotherapy'
os.chdir (dataFolderPath)

df_immuno_all = pd.read_csv('Immunotherapy.csv', sep=',') # adult data training set (including the validation set)

immuno_all = df_immuno_all.to_numpy()
# labels 
immuno_labels = immuno_all[:,7]
# design matrix
immuno_desMat = immuno_all[:,0:7]

# splittting data
xTrain_im, xTest_im, yTrain_im, yTest_im = train_test_split(immuno_desMat, immuno_labels, test_size=0.2, random_state = 0)

# k-fold cross validation 

x_im = xTrain_im
y_im = yTrain_im
# convert the label from string ' <=50K' or ' >50K' into binary numbers
# probably don't need to convert labels into strings 

xy_conc_im= np.column_stack((x_im,y_im)) # cannot use np.concatenate of data with different dimensions 
dataset_im = xy_conc_im; folds = 5 # delete folds later when embedded in function input 
folds = 5
dataset_split_im = kfold_cross_validation(dataset_im,folds)

# delete the last column of each of the five fol, which is the label, while storing the label
cv_train_data_im = []
cv_train_label_im = []; label_ind_im = len(xy_conc_im[1,:])-1
for i in range (folds): 
    # access the last label column of matrix 
    aa = dataset_split_im[i]
    bb = np.array(aa)
    label = bb [:,label_ind_im]
    label_array = np.array(label)
    cv_train_label_im.append(label)
    # delete the last label column of each matrix fold 
    b = np.delete(dataset_split_im[i],label_ind_im,1) # 1 means column 
    cv_train_data_im.append(b)
    
# cv_train_data has k (5) folds including the 4 folds for training set and 1 fold for testing set 
# cv_train_label has k (5) folds for the corresponding labels for the 5 folds in cv_train_data
    
# training the model k times so change the index of training_data and training_labels 

training_data_im = cv_train_data_im[0]
training_labels_im = cv_train_label_im[0]


## LOGISTIC REGRESSION MODEL IMPLEMENTATION AND TESTING 
dataFolderPath = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Models'
os.chdir (dataFolderPath)

import logisticRegression as lr
  
immuno_lr = lr.Logistic_Regression(0.02,"immunotherapy","binary")
params = lr.fit(immuno_lr,training_data_im, training_labels_im, 0.01, 1e-2)




