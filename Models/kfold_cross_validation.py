# -*- coding: utf-8 -*-
"""
K-fold cross validation:
    5-fold cross validation to estimate performance in all experiments
    evaluate the performance using accuracy 
    
    tasks: 
        1. Compare the accuracy of naive Bayes and logistic regression on the four datasets
        2. Test different learning rates for gradient descent appleid to logistic regression
        Use the threshold for change in the value of the cost function as termination criteria, 
        and plot the accuracy on train/validation set as a function of iterations of gradient descent. 
        3.  Compare the accuracy of the two models as a function of the size of dataset (by controlling the training size). 
        As an example, see Figure 1 here 1. 
"""


from random import randrange
import numpy as np
from sklearn.model_selection import train_test_split
### splitting data into training and testing with a 80:20 ratio (train:test), train includes validation 
# only use sklearn to split, execute the following 
# x = features_all_np
# y = labels_all_np
def split_train_test (x,y): 

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
# xTrain = training instances + validation instances = 34368 instances in total, 108 features 
# yTrain = labels for xTrian 
# xTest = testing instances = 8592 instances, 108 features 
# yTest = labels for yTest 
# xTrain now undergo k-fold cross validation 

### k-fold cross validation 
# https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/

# We calculate the size of each fold as the size of the dataset divided by the number of folds required.

total_rows = len (x); total_folds = 5; 
fold_size = total_rows / total_folds
# If the dataset does not cleanly divide by the number of folds, 
# there may be some remainder rows and they will not be used in the split.
# We then create a list of rows with the required size and add them to a list of folds which is 
# then returned at the end.
# Split a dataset into k folds
from random import seed
from random import randrange
dataset = x; folds = 5 # delete folds later when embedded in function input 

def kfold_cross_validation(dataset, folds=5):
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
    '''
    dataset_split=list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
	    fold = list()
	    while len(fold) < fold_size:
		    index = randrange(len(dataset_copy))
		    fold.append(dataset_copy.pop(index))
		    dataset_split.append(fold)
    return dataset_split



scores_logisticReg = []
scores_niaveBayes = []

# for train_index, test_index in folds.split(digits.data,digits.target):
#     X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
#                                        digits.target[train_index], digits.target[test_index]
#     scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
#     scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
#     scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
    
# back to preprocessing, concatenating the training and testing sets 




