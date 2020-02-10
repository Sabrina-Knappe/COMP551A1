# -*- coding: utf-8 -*-

"""
    In addtion to model classes, 
    
    define a function evaluate_acc to evaluate the model accuracy
        
        This function takes the true labels (i.e. y) and the target labels (i.e. y_hat) as input
        and it should output the accuracy score
        
    define a function to run k-fold cross validation 
"""

def evaluate_acc(y,y_hat):
    '''
    input: 
        y: true labels  
        y_hat: target labels 
        
    output:
        accuracy score of the models 
    '''
# Accuracy, Recall, Precision, F1 Score in Python from scratch
    # https://www.youtube.com/watch?v=9PbrWiLC-4k
    
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae



### classification metrics - Accuracy, Recall, Precision, F1 Score
# https://github.com/bhattbhavesh91/classification-metrics-python/blob/master/ml_a.ipynb
    

### k-fold cross validation 
# https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/

# We calculate the size of each fold as the size of the dataset divided by the number of folds required.

fold size = total rows / total folds
# If the dataset does not cleanly divide by the number of folds, 
# there may be some remainder rows and they will not be used in the split.
# We then create a list of rows with the required size and add them to a list of folds which is 
# then returned at the end.
from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
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

from random import seed
from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
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

# test cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)    





