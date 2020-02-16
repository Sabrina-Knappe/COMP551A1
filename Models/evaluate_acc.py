# -*- coding: utf-8 -*-

"""
    In addtion to model classes, 
    
    define a function evaluate_acc to evaluate the model accuracy
        
        This function takes the true labels (i.e. y) and the target labels (i.e. y_hat) as input
        and it should output the accuracy score
        
    define a function to run k-fold cross validation 
"""
import numpy as np 
from scipy.integrate import simps
from numpy import trapz
import matplotlib.pyplot as plt
import sympy as sy

      
    

def evaluate_acc(y_act,y_pred):
    '''
    input: 
        y_act: true labels  
        y_pred: predicted labels 
        
    output:
        various types of accuracy scores of the models 
    '''
# Accuracy, Recall, Precision, F1 Score in Python from scratch
    # https://www.youtube.com/watch?v=9PbrWiLC-4k
    tp,tn,fp,fn = compute_tp_tn_fn_fp(y_act,y_pred)
    acc_score=compute_accuracy(tp,tn,fp,fn)
    precision_score=compute_precision(tp, fp)
    recall_score=compute_recall(tp, fn)
    f1_score=compute_f1_score(y_true, y_pred)
    print('Accuracy Score:',acc_score);
    print('Precision Score:',precision_score)
    print('Recall Score:',recall_score)
    print('f1 Score:',f1_score)
        
    return acc_score,precision_score,recall_score,f1_score

    

def compute_tp_tn_fn_fp(y_act, y_pred):
    
    
    tp=0; tn=0; fn=0;fp=0
    for i in range(len(y_act)):
        if y_act[i] == 1 and y_pred[i] == 1:
            tp+=1
        elif y_act[i] == 0 and y_pred[i] == 0:
            tn+=1
        elif y_act[i] == 0 and y_pred[i] == 1:
            fn+=1
        elif y_act[i] == 1 and y_pred[i] == 0:
            fp+=1
    
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

def ROC(tp,fn,fp,tn):
    '''
    ROC Cruve Value 
    https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) 
    for a number of different candidate threshold values between 0.0 and 1.0
    
    1) The true positive rate is calculated as the number of true positives divided by 
    the sum of the number of true positives and the number of false negatives. 
    It describes how good the model is at predicting the positive class when the actual outcome is positive.

    True Positive Rate = True Positives / (True Positives + False Negatives)
    also called Sensitivity  = True Positives / (True Positives + False Negatives)
    
    2)  The false positive rate is calculated as the number of false positives divided by 
    the sum of the number of false positives and the number of true negatives.
    It is also called the false alarm rate as it summarizes how often a positive class is predicted when the actual outcome is negative.

    False Positive Rate = False Positives / (False Positives + True Negatives)
    also called Inverted Specificity 
    Specificity = True Negatives / (True Negatives + False Positives)
    False Positive Rate = 1 - Specificity
    
    3) reason to use ROC 
    The curves of different models can be compared directly in general or for different thresholds.
    The area under the curve (AUC) can be used as a summary of the model skill.
    
    4) Hw to interpret ROC and its AUC Values 
    Smaller values on the x-axis of the plot indicate lower false positives and higher true negatives.
    Larger values on the y-axis of the plot indicate higher true positives and lower false negatives.
    
        best ROC has an AUC of 1; worse ROC has an AUC of 0.5 
    
    '''
    # iterate through an array of threshold points 
     # True Position Rate 
    TPR = tp/(tp+fn) # SENSITIVITY, y-aixs
    FPR = fp/(fp+tn) # 1- SPECIFICITY, x-aixs 
    
    return TPR, FPR

def compute_AUC (threshold, TPR, FPR):
    
    # compute AUC (Area Under the Curve) value of the ROC curve
    # TPR,FPR = ROC(tp,fn,fp,tn)
    auc = trapz(FPR,TPR,dx=0.1)
    
    # ONE TRESHOLD VALUE PRODUCES ONE (FPR,TPR) data point and then the ROC curve is 
    # composed of all points with a different threshold 
    return auc


# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


def train_random_sample(perctage,whole_train_data):
    '''
    percentage: between 0 and 1
    whole_train_data: whole train dataset including the labels, do k-fold CV outside of this function 
    
    train_random: randomly selected train set with asked percentage of the whole train set, including labels
    
    '''
    # use random.sample to randomly draw indices and use that indices to extract 
    # the train data and its label with the same index
    num2draw = int(percentage*len(whole_train_data))
    whole_ind = np.arange(0,num2draw,1).tolist()
    draw_ind = random.sample(whole_ind,k=num2draw)
    train_random = whole_train_data[draw_ind,:]
    
    return train_random

