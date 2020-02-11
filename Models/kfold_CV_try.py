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
    
    cv_train_data: k-fold; 
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
    
    # delete the last column of each of the five fol, which is the label, while storing the label
    cv_train_data = []
    cv_train_label = []; label_ind = len(dataset[1,:])-1
    for i in range (folds): 
        # access the last label column of matrix 
        aa = dataset_split[i]
        bb = np.array(aa)
        label = bb [:,label_ind]
        label_array = np.array(label)
        cv_train_label.append(label)
        # delete the last label column of each matrix fold 
        b = np.delete(dataset_split[i],label_ind,1) # 1 means column 
        cv_train_data.append(b)
        
    return dataset_split,cv_train_data,cv_train_label

def train_validation_split(cv_train_data,cv_train_label,fold_num):
    '''
    input:
        cv_train_data: k-folded (k=5 here) training data to be split into 
            k-1 folds for training set
            1 fold for testing set
        cv_train_label: cv_train_data corresponding labels 
        fold_num: the number of time you are evaluating 
    '''
    if fold_num == 1:
        validate_data = cv_train_data[0]
        validate_labels = cv_train_label[0]
        training_data = np.concatenate((cv_train_data[1], cv_train_data[2],cv_train_data[3],cv_train_data[4]))
        training_labels = np.concatenate((cv_train_label[1], cv_train_label[2],cv_train_label[3],cv_train_label[4]))

    if fold_num ==2:
        validate_data = cv_train_data[1]
        validate_labels = cv_train_label[1]
        training_data = np.concatenate((cv_train_data[0], cv_train_data[2],cv_train_data[3],cv_train_data[4]))
        training_labels = np.concatenate((cv_train_label[0], cv_train_label[2],cv_train_label[3],cv_train_label[4]))
        
    if fold_num ==3:
        validate_data = cv_train_data[2]
        validate_labels = cv_train_label[2]
        training_data = np.concatenate((cv_train_data[1], cv_train_data[0],cv_train_data[3],cv_train_data[4]))
        training_labels = np.concatenate((cv_train_label[1], cv_train_label[0],cv_train_label[3],cv_train_label[4]))
        
    if fold_num ==4:
        validate_data = cv_train_data[3]
        validate_labels = cv_train_label[3]
        training_data = np.concatenate((cv_train_data[1], cv_train_data[2],cv_train_data[0],cv_train_data[4]))
        training_labels = np.concatenate((cv_train_label[1], cv_train_label[2],cv_train_label[0],cv_train_label[4]))
        
    if fold_num ==5: 
        validate_data = cv_train_data[4]
        validate_labels = cv_train_label[4]
        training_data = np.concatenate((cv_train_data[1], cv_train_data[2],cv_train_data[3],cv_train_data[0]))
        training_labels = np.concatenate((cv_train_label[1], cv_train_label[2],cv_train_label[3],cv_train_label[0]))
        
    validate_data=validate_data.astype(object)
    validate_labels=validate_labels.astype(object)
    training_data=training_data.astype(object)
    training_labels=training_labels.astype(object)
    
    return validate_data,validate_labels,training_data,training_labels
        
    
    
    
    
