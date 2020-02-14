# -*- coding: utf-8 -*-
"""
Use Sklearn to get some results 
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# NAIVE BAYES MODEL FOR DATASET Immunotherapy
import urllib 
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
# from sklearn.cross_validation import train_test_split 
# LOGISTIC REGRESSION MODEL 
from pandas import Series, DataFrame 
import scipy
from scipy.stats import spearmanr

from pylab import rcparams
import seaborn as sb 
import matplotlib.pyplot as plt 

import sklearn
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 



from sklearn import metrics 

val_acc_score_Ber=[]; val_acc_score_Multi=[]; val_acc_score_gau=[]
for i in range(5):
    
    validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,(i+1))

    # TEST BERNOULLI NAIVE BAYES on trainng and validation sets 
    del BernNB; del MultiNB; del GausNB
    BernNB = BernoulliNB(binarize = True)
    BernNB.fit(training_data_im,training_labels_im)
    print(BernNB)
    validate_expect = validate_labels_im
    validate_pred= BernNB.predict(validate_data_im)
    # BernulliNB accuracy score 
    validate_acc_score_ber = accuracy_score(validate_expect,validate_pred)
    print(validate_acc_score_ber)
    val_acc_score_Ber.append(validate_acc_score_ber)
    
    # Multinomail NAIVE BAYES 
    MultiNB = MultinomialNB()
    MultiNB.fit(training_data_im,training_labels_im)
    print(MultiNB)
    validate_expect = validate_labels_im
    validate_pred= MultiNB.predict(validate_data_im)
    # MultinomialNB accuracy score 
    validate_acc_score_mul = accuracy_score(validate_expect,validate_pred)
    print(validate_acc_score_mul)
    val_acc_score_Multi.append(validate_acc_score_mul)
    
    # Gaussian NAIVE BAYES 
    GausNB = GaussianNB()
    GausNB.fit(training_data_im,training_labels_im)
    print(GausNB)
    validate_expect = validate_labels_im
    validate_pred= GausNB.predict(validate_data_im)
    # MultinomialNB accuracy score 
    validate_acc_score_gau = accuracy_score(validate_expect,validate_pred)
    print(validate_acc_score_gau)
    val_acc_score_gau.append(validate_acc_score_gau)

mu_NB_Ber=np.mean(val_acc_score_Ber); mu_NB_Mul=np.mean(val_acc_score_Multi); 
mu_NB_gau=np.mean(val_acc_score_gau)
print('Bernoulli NB accuracy score:',mu_NB_Ber)
print('Multinomial NB accuracy score:',mu_NB_Mul)
print('Gaussian NB accuracy score:',mu_NB_gau)
# CHOOSE ONE OF BEST THE CLASSES (BERNOULLI,MULTINOMIAL,GAUSSIAN) WITH THE HIGHEST ACCURACY SCORE

# BERNOULI NAIVE BAYES IS THE BEST PERFORMING MODEL for this dataset based on the mean cross validation 
# of the immunotherapy dataset, now we may calculate the precision, recall and f1 scores of the Bernulli model
vali_acc_score_nb=[]; vali_preci_socre_nb=[]; vali_recall_score_nb=[];vali_f1_score_nb=[]
for i in range(5):
    
    validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,(i+1))

    # TEST BERNOULLI NAIVE BAYES on trainng and validation sets 
    del BernNB; 
    BernNB = BernoulliNB(binarize = True); BernNB.fit(training_data_im,training_labels_im)
    validate_expect = validate_labels_im; validate_pred= BernNB.predict(validate_data_im)
    # BernulliNB accuracy score 
    validate_acc_score_ber = accuracy_score(validate_expect,validate_pred)
    vali_acc_score_nb.append(validate_acc_score_ber)
    # precision score
    validate_preci_score_nb=precision_score(validate_expect,validate_pred)                     
    vali_preci_socre_nb.append(validate_preci_score_nb)    
    # recall score 
    validate_recall_score_nb = recall_score(validate_expect,validate_pred)
    vali_recall_score_nb.append(validate_recall_score_nb)
    # f1 score
    validate_f1_score_lr = f1_score(validate_expect,validate_pred) 
    vali_f1_score_nb.append(validate_f1_score_lr)
mu_vali_acc_nb=np.mean(vali_acc_score_nb); 
mu_vali_preci_nb=np.mean(vali_preci_socre_nb); 
mu_vali_recall_nb=np.mean(vali_recall_score_nb);
mu_vali_f1_nb=np.mean(vali_f1_score_nb)
print('Ber NB accuracy score:',mu_vali_acc_nb)
print('Ber NB precision score:',mu_vali_preci_nb)
print('Ber NB recall score:',mu_vali_recall_nb)
print('Ber NB f1 score:',mu_vali_f1_nb)

###############################################################################
# LOGISTIC REGRESSION MODEL 
# APPEND SCORES
vali_acc_score_lr=[]; vali_preci_socre_lr=[]; vali_recall_score_lr=[];vali_f1_score_lr=[]
# implement LR model 
for i in range(5):
    
    validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,(i+1))
    # delete the previous loop's model 
    del LogReg
    LogReg =LogisticRegression()
    LogReg.fit(training_data_im,training_labels_im)
    
    # measure accuracy of model LogReg using validation set 
    LogReg.score(validate_data_im,validate_labels_im)
    
    # LogReg.predict_proba(validate_data_im)
    validate_pred_lr=LogReg.predict(validate_data_im)
    
    # accuracry score:
    validate_expect_lr = validate_labels_im
    validate_acc_score_lr = accuracy_score(validate_expect_lr,validate_pred_lr)
    vali_acc_score_lr.append(validate_acc_score_lr)
    # confusion matrix 
    tn_lr1, fp_lr1, fn_lr1, tp_lr1 = confusion_matrix(validate_expect_lr,validate_pred_lr).ravel()
    
    #print('TP for LR:', tp_lr1)
    #print('TN for LR :', tn_lr1)
    #print('FP for LR :', fp_lr1)
    #print('FN for LR :', fn_lr1)
    
    # precision score
    validate_preci_score_lr=precision_score(validate_expect_lr,validate_pred_lr)                     
    vali_preci_socre_lr.append(validate_preci_score_lr)    
    # recall score 
    validate_recall_score_lr = recall_score(validate_expect_lr,validate_pred_lr)
    vali_recall_score_lr.append(validate_recall_score_lr)
    # f1 score
    validate_f1_score_lr = f1_score(validate_expect_lr,validate_pred_lr) 
    vali_f1_score_lr.append(validate_f1_score_lr)

mu_vali_acc_lr=np.mean(vali_acc_score_lr); 
mu_vali_preci_lr=np.mean(vali_preci_socre_lr); 
mu_vali_recall_lr=np.mean(vali_recall_score_lr);
mu_vali_f1_lr=np.mean(vali_f1_score_lr)
print('LR accuracy score:',mu_vali_acc_lr)
print('LR precision score:',mu_vali_preci_lr)
print('LR recall score:',mu_vali_recall_lr)
print('LR f1 score:',mu_vali_f1_lr)

##############################################################################
# QUESTION 3-PART2: CHANGING LEARNING RATE ON THE LOGISTIC REGRESSION MODEL 








###############################################################################
# QUESTION 3 - PART 3: 
import random

# calculate the number of elemets to randomly draw before random sampling 
# THE TOTAL TRAINING SET DATA 
whole_train_data=xy_conc_im
percentage=0.85  # 100%, randomly drawn 85%, 70%, 65% 

# just run code inside function 
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
    
# 5-fold cross validatoion splitting from train_random 
folds=5
# train_random=train_random_sample(0.75,whole_train_data)
dataset_split_trainRn, cv_train_data_trainRn,cv_train_label_trainRn= kfold_cross_validation(train_random,folds)

# re-run the LR model on this n% training set, and test on the test set and produce an accuracy score 
# test data from data_preprocess_immuno: xTrain_im, xTest_im, yTrain_im, yTest_im = train_test_split(immuno_desMat, immuno_labels, test_size=0.2, random_state = 0)

test_acc_score_lr_rn=[]
for i in range(folds):
    
    validate_data_rn,validate_labels_rn,training_data_rn,training_labels_rn = train_validation_split(cv_train_data_trainRn,cv_train_label_trainRn,(i+1))
    # delete the previous loop's model 
    del LogReg_rn
    LogReg_rn =LogisticRegression()
    LogReg_rn.fit(training_data_rn,training_labels_rn)
    
    # measure accuracy of model LogReg using validation set 
    LogReg_rn.score(validate_data_rn,validate_labels_rn)
    # LogReg.predict_proba(validate_data_im)
    test_pred_lr_rn=LogReg_rn.predict(xTest_im)
    # accuracry score:
    test_expect_lr_rn = yTest_im
    test_acc_score_lr = accuracy_score(test_expect_lr_rn,test_pred_lr_rn)
    test_acc_score_lr_rn.append(test_acc_score_lr)
    
mu_test_acc_score_lr_rn=np.mean(test_acc_score_lr_rn)
print('LR avg accuracy score of',percentage,'dataset:',mu_test_acc_score_lr_rn)






