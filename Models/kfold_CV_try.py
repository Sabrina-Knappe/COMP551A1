# -*- coding: utf-8 -*-
"""
splitting test and train data, assumed a 80:20 ratio, can change it inside the code! 

k-fold cross validation, here k =5, returns variabel "dataset_split" which is 
    five chunks of your training data fed into it (i.e. train+validation data)
"""

from random import randrange
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange

# x = features_all_np
# y = labels_all_np
# test_size = 0.2
def split_train_test (x,y,test_size): 

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state = 0)
    return xTrain,xTest,yTrain,yTest
# xTrain = training instances + validation instances = 34368 instances in total, 108 features 
# yTrain = labels for xTrian 
# xTest = testing instances = 8592 instances, 108 features 
# yTest = labels for yTest 
# xTrain now undergo k-fold cross validation 



# k-fold cross validation, execute the following 
# x = xTrain
# y = yTrain 
# xy_conc = np.column_stack((x,y)) # cannot use np.concatenate of data with different dimensions 
# dataset = xy_conc; folds = 5 # delete folds later when embedded in function input 
# folds = 5

def kfold_cross_validation(dataset, folds):
    '''
    Generally, you split data into training-validation-test sets. 
    The goal of cross-validation (CV) is to validate your model multiple times 
    so CV is not related to test set and only related to training/validation set. 
    When doing CV, you split the training+validation set into k-folds and 
    each time (k times in total) you take k-1 folds as training set and 1 fold 
    as validation set. So you will get k training acc and k validation acc. 
    You compare the averaged accuracy among different models or different hyperparameters. 
    After CV, you should use training+validation (e.g. all the k-folds) set to retrain 
    your model and test on test set to get the final test accuracy. Notice the 
    test set is only used once.
    
    dataset_split: training set split into k fold, including the last column 
    to be its corresponding label; 
    k-1 will become the training set; 
    1 fold becomes the validatio set 
    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
	    fold = list()
	    while len(fold) < fold_size:
		    index = randrange(len(dataset_copy))
		    fold.append(dataset_copy.pop(index))
	    dataset_split.append(fold)

    return dataset_split
