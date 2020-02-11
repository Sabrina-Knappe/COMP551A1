# -*- coding: utf-8 -*-

"""
    In addtion to model classes, 
    
    define a function evaluate_acc to evaluate the model accuracy
        
        This function takes the true labels (i.e. y) and the target labels (i.e. y_hat) as input
        and it should output the accuracy score
        
    define a function to run k-fold cross validation 
"""
import numpy as np 

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

def compute_tp_tn_fn_fp(y_act, y_pred):
	'''
    input:
        y_act = actual label 
        y_pred = predicted label 
    output: 
    	True positive - actual = 1, predicted = 1
    	False positive - actual = 1, predicted = 0
    	False negative - actual = 0, predicted = 1
    	True negative - actual = 0, predicted = 0
	'''
	tp = sum((y_act == 1) & (y_pred == 1))
	tn = sum((y_act == 0) & (y_pred == 0))
	fn = sum((y_act == 1) & (y_pred == 0))
	fp = sum((y_act == 0) & (y_pred == 1))
	return tp, tn, fp, fn

def compute_accuracy(tp, tn, fn, fp):
	'''
	Accuracy = TP + TN / FP + FN + TP + TN

	'''
	return ((tp + tn) * 100)/ float( tp + tn + fn + fp)

# to test 
# print('Accuracy for Logistic Regression :', compute_accuracy(tp_lr, tn_lr, fn_lr, fp_lr))
# print('Accuracy for Niave Bayes :', compute_accuracy(tp_nb, tn_nb, fn_nb, fp_nb))
    
def compute_precision(tp, fp):
	'''
	Precision = TP  / FP + TP 

	'''
	return (tp  * 100)/ float( tp + fp)
# to test 
# print('Precision for Logistic Regression :', compute_precision(tp_lr, fp_lr))
# print('Precision for Niave Bayes :', compute_precision(tp_nb, fp_nb))

def compute_recall(tp, fn):
	'''
	Recall = TP /FN + TP 

	'''
	return (tp  * 100)/ float( tp + fn)
# to test 
# print('Recall for Logistic Regression :', compute_recall(tp_lr, fn_lr))
# print('Recall for Niave Bayes :', compute_recall(tp_nb, fn_nb))

def compute_f1_score(y_true, y_pred):
    # calculates the F1 score
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y_true, y_pred)
    precision = compute_precision(tp, fp)/100
    recall = compute_recall(tp, fn)/100
    f1_score = (2*precision*recall)/ (precision + recall)
    return f1_score
# to test 
# print('F1 score for Logistic Regression :', compute_f1_score(df.y_act, df.y_pred_lr))
# print('F1 score for Niave Bayes:', compute_f1_score(df.y_act, df.y_pred_nb))

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
# def fit_and_evaluate(model):
    
    # Train the model
#    model.fit(X_train, y_train)
    
    # Make predictions and evalute
#    model_pred = model.predict(X_test)
#    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
#    return model_mae



