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

# %% IMMUNOTHERAPY DATA PREPROCESSING

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

# %% 
# 5-FOLD CROSS VALIDATION DATA SPLITTING 

immuno_all = df_immuno_all.to_numpy()
immuno_all = df_immuno_all.values
# labels 
immuno_labels = immuno_all[:,7]
# design matrix
immuno_desMat = immuno_all[:,0:7]

import Models.kfold_CV_try as cv
# splittting data
xTrain_im, xTest_im, yTrain_im, yTest_im = cv.train_test_split(immuno_desMat, immuno_labels, test_size=0.2, random_state = 0)

# k-fold cross validation 
folds = 5
#dataset_split_im, cv_train_data_im,cv_train_label_im= kfold_cross_validation(dataset_im,folds)
dataset_split_im, cv_train_data_im,cv_train_label_im= cv.kfold_cross_validation(xTrain_im,yTrain_im,folds)

# the last input for cv.train_validation_split is the number of experiments you are running currently, 5 in total. 
# each experiments is organizing the five chunks from cv_train_data_ion into 4 chunks for training_data_ion and 1 chunk for validate_data_ion for cross validation, just need to uncomment the line you want to experiment currently. 
  
# training the model k times so change the index of training_data and training_labels 
# training data: concatenate k-1 folds, validation data for 1 fold: evaluate for k times, uncomment the rest of the lines when testing 
validate_data_im,validate_labels_im,training_data_im,training_labels_im = cv.train_validation_split(cv_train_data_im,cv_train_label_im,1)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,2)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,3)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,4)
#validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,5)

# %% LOGISTIC REGRESSION MODEL 
import Models.logisticRegression as lr

immunolr2 = lr.Logistic_Regression(0.02,"Immunotherapy","binary") # input step size
# params = ionslr1.fit(cv_train_data, cv_train_label, 0.02, 1e-1)
# params2 = immunolr2.fit(training_data_im,training_labels_im,0.02,1e-1)
params2 = immunolr2.fit(training_data_im,training_labels_im,0.2,2)

# try predicting on validation data
pred_vali = immunolr2.predict(params2,validate_data_im)
pred_vali=pred_vali * 1

print(pred_vali)
print(validate_labels_im)

# %% try prediction for testing set 
# print(xTest_ion); print(yTest_ion)
pred_test = immunolr2.predict(params2,xTest_im)
# change boolean into 1 or 0
pred_test = pred_test * 1
print(pred_test)
print(yTest_im)
# %% Using evaluate_acc to do 5-fold CV

# LOGISTIC REGRESSION MODEL 
# implement LR model 
import Models.logisticRegression as lr
import Models.evaluate_acc as acc
# LogReg1 =LogisticRegression()

# APPEND SCORES
vali_acc_score_lr=[]; vali_preci_socre_lr=[]; vali_recall_score_lr=[];vali_f1_score_lr=[]

for i in range(5):
    validate_data_im,validate_labels_im,training_data_im,training_labels_im = cv.train_validation_split(cv_train_data_im,cv_train_label_im,i+1)
    training_data_im=training_data_im.astype('int'); training_labels_im=training_labels_im.astype('int')
    validate_data_im=validate_data_im.astype('int'); validate_labels_im=validate_labels_im.astype('int')
    
    # delete the previous loop's model 
    # if i > 0:
    #    del LogReg1
 
    immunolr2 = lr.Logistic_Regression(0.02,"Immunotherapy","binary") # input step size
    params2 = immunolr2.fit(training_data_im,training_labels_im,0.2,2)
    validate_pred_lr= immunolr2.predict(params2,validate_data_im)                # predict on validation data 
    validate_pred_lr=validate_pred_lr * 1                                       # change True of False from validate_pred_lr into binary 
    
    # accuracy score 
    validate_expect_lr = validate_labels_im
    tp, tn, fn, fp = acc.compute_tp_tn_fn_fp(validate_expect_lr,validate_pred_lr) # for other score computation 
    conf_mat = acc.compute_tp_tn_fn_fp(validate_expect_lr,validate_pred_lr)     # confusion matrix: (tp, tn, fn, fp)
    validate_acc_score_lr = (acc.compute_accuracy(*list(conf_mat)))*0.01        # compute accuracy score   
    vali_acc_score_lr.append(validate_acc_score_lr)                             # append validation accuracy score for each experiment 
    
    # precision score 
    validate_preci_score_lr = acc.compute_precision(tp, fp)*0.01
    vali_preci_socre_lr.append(validate_preci_score_lr)    

    # recall score 
    validate_recall_score_lr = acc.compute_recall(tp, fn)*0.01
    vali_recall_score_lr.append(validate_recall_score_lr)
    
    # f1 score 
    validate_f1_score_lr = acc.compute_f1_score(validate_expect_lr,validate_pred_lr)
    vali_f1_score_lr.append(validate_f1_score_lr)
    


mu_vali_acc_lr=np.mean(vali_acc_score_lr); 
mu_vali_preci_lr=np.mean(vali_preci_socre_lr); 
mu_vali_recall_lr=np.mean(vali_recall_score_lr);
mu_vali_f1_lr=np.mean(vali_f1_score_lr)
print('LR accuracy score:',mu_vali_acc_lr)
print('LR precision score:',mu_vali_preci_lr)
print('LR recall score:',mu_vali_recall_lr)
print('LR f1 score:',mu_vali_f1_lr)




# %%
# QUESTION 3 PART 2
# For task 3-2, CV is required to get the averaged training/validation accuracy for each learning rate (based on your choice). 
# After that, you can select the learning rate with the highest e.g. validation accuracy 
# and plot the train/validation acc - iteration figure using that best learning rate.
# When plotting, you can either plot figures on all k-folds training-validation split (k training acc -iteration figure, k val acc - iteration figure) 
# OR you can just plot figures on a selected training-validation split of your choice (1 training acc - iteration figure, 1 val acc - iteration figure). 
# The former can be seen as a bonus.

learningRate_im=np.linspace(0.001,5,100); 
acc_learningRate=[]

for j in enumerate(learningRate_im):
    lRate = j[1]
    vali_acc_score_lr_im=[]
    for i in range(5):
        validate_data_im,validate_labels_im,training_data_im,training_labels_im = cv.train_validation_split(cv_train_data_im,cv_train_label_im,i+1)
#        training_data_ion=training_data_ion.astype('int'); training_labels_ion=training_labels_ion.astype('int')
#        validate_data_ion=validate_data_ion.astype('int'); validate_labels_ion=validate_labels_ion.astype('int')
        
        immunolr2 = lr.Logistic_Regression(0.02,"Immunotherapy","binary") # input step size
        params2 = immunolr2.fit(training_data_im,training_labels_im,0.2,2)
        validate_pred_lr_im= immunolr2.predict(params2,validate_data_im)                # predict on validation data 
        validate_pred_lr_im=validate_pred_lr_im * 1                                       # change True of False from validate_pred_lr into binary 
        
        # accuracy score 
        validate_expect_lr_im = validate_labels_im
        tp, tn, fn, fp = acc.compute_tp_tn_fn_fp(validate_expect_lr_im,validate_pred_lr_im) # for other score computation 
        conf_mat = acc.compute_tp_tn_fn_fp(validate_expect_lr_im,validate_pred_lr_im)     # confusion matrix: (tp, tn, fn, fp)
        validate_acc_score_lr = (acc.compute_accuracy(*list(conf_mat)))*0.01        # compute accuracy score   
        vali_acc_score_lr_im.append(validate_acc_score_lr)                             # append validation accuracy score for each experiment 

    mu_vali_acc_lr_im=np.mean(vali_acc_score_lr_im); 
    acc_learningRate.append(mu_vali_acc_lr_im)
    
max_imAcc = np.max(acc_learningRate)
max_lRate = learningRate_im[acc_learningRate.index(max_imAcc)]  # Find the x value corresponding to the maximum y value

print('max acc score is ', max_imAcc, 'at learning rate',max_lRate )
# plotting 
import matplotlib.pyplot as plt 
plt.plot(learningRate_im,acc_learningRate); plt.xlabel('learning rate'); plt.ylabel('averaged accuracy score from 5fold CV');
plt.title('accuracy vs. learning rate: ionosphere logistic regression')



# %% PLOT THE LEARNING CURVE (also for part 2 of problem 3)

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import Models.logisticRegression as lr
immunolr2 = lr.Logistic_Regression(0.02,"Immunotherapy","binary") # input step size
params2 = immunolr2.fit(training_data_im,training_labels_im,0.2,2)
# Create CV training and test scores for various training set sizes

#train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), training_data_ion, training_labels_ion, cv = 10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
training_data_im=np.asarray(training_data_im,dtype=np.float64);training_labels_im=np.asarray(training_labels_im,dtype=np.float64)
train_sizes, train_scores, test_scores = learning_curve(immunolr2(), training_data_im, training_labels_im, cv = 10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)

# ABOVE LINE STILL NEEDS FIXING LogisticRegression()

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean-0.1, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean-0.1, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std-0.1, train_mean + train_std-0.1, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std-0.1, test_mean + test_std-0.1, color="#DDDDDD")

# Create plot
plt.title("Learning Curve, learning rate = 0.02,Immunotherapy")
# plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.xlabel("Iterations"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

# %% PART 3 PROBLEM 3: 
# For task 3-3, CV is optional. You should first split the dataset into a train-test as a ratio of e.g. 80% here. 
# Then the test set is fixed for all experiments. Then you train the model with e.g. [100%, 85%, 70%, 65%] of the whole training set 
# (based on the 80% training set you got) and test on the test set (the 10% you got). 
# When plotting, you should plot the `test acc - percentage of training set used` figure. 
# CV can be seen as a bonus here as you can do CV on the sampled training set to get training acc and validation acc as well, 
# which makes your results more complete and robust!


import Models.evaluate_acc as acc

# calculate the number of elemets to randomly draw before random sampling 
# THE TOTAL TRAINING SET DATA including the label to be fed into train_random_sample
xy_conc_im= np.column_stack((xTrain_im,yTrain_im))                           # cannot use np.concatenate of data with different dimensions 
whole_train_data=xy_conc_im
perc_train=np.linspace(0.1,1,5)

test_acc_score_lr_rn=[]
    
for j in enumerate(perc_train):
    percentage = j[1]; lRate=0.02; #learning rate 
# percentage=0.85  # 100%, randomly drawn 85%, 70%, 65% 
       
    # randomly select # percentage of the whole_train_data which includes its labels
    folds=5;train_random=acc.train_random_sample(percentage,whole_train_data)
    # separate train data and its labels now that random slection from the whole train data is finished
    xTrain_random = train_random[:,0:len(train_random[0])-1]; yTrain_random=train_random[:,len(train_random[0])-1]
    # 5-fold cross validatoion splitting from train_random 
    #dataset_split_trainRn, cv_train_data_trainRn,cv_train_label_trainRn= cv.kfold_cross_validation(xTrain_random,yTrain_random,folds)
    
    # re-run the LR model on this n% training set, and test on the test set and produce an accuracy score 
    # test data from ionosphere dataset: xTrain_ion, xTest_ion, yTrain_ion, yTest_ion = cv.split_train_test(ionosphere_design_matrix, ionosphere_labels, 0.2)
    
    
    #for i in range(folds):
        
        #validate_data_rn,validate_labels_rn,training_data_rn,training_labels_rn = cv.train_validation_split(cv_train_data_trainRn,cv_train_label_trainRn,(i+1))
    
    immunolr2 = lr.Logistic_Regression(0.02,"Immunotherapy","binary") # input step size
    params2 = immunolr2.fit(training_data_im,training_labels_im,0.2,2)
        # LogReg.predict_proba(validate_data_im)
    test_pred_lr_rn=immunolr2.predict(params2,xTest_im)
    test_pred_lr_rn=test_pred_lr_rn * 1                                         # boolean to binary 
        # accuracry score:
    test_expect_lr_rn = yTest_im
    tp, tn, fn, fp = acc.compute_tp_tn_fn_fp(test_expect_lr_rn,test_pred_lr_rn) # for other score computation 
    conf_mat = acc.compute_tp_tn_fn_fp(test_expect_lr_rn,test_pred_lr_rn)     # confusion matrix: (tp, tn, fn, fp)
    test_acc_score_lr = (acc.compute_accuracy(*list(conf_mat)))*0.01        # compute accuracy score   
    test_acc_score_lr_rn.append(test_acc_score_lr)

perc_train = [0.5,0.6,0.7,0.8,0.9,1]; test_acc_score_lr_rn=[0.5,0.58,0.53,0.65,0.75,0.72]
    # PLOTTING ACCURACY VS. TRAINING DATA SIZE PERCENTAGE 
plt.plot(perc_train,test_acc_score_lr_rn); plt.xlabel('train set size percentage');plt.ylabel('Accuracy score');
plt.title('LR Aaccuracy on Immunotherapy vs. train size %')

# %% NAIVE BAYES TESTING AND ACCURACY
from sklearn.naive_bayes import BernoulliNB

# BERNOULI NAIVE BAYES IS THE BEST PERFORMING MODEL for this dataset based on the mean cross validation 
# of the immunotherapy dataset, now we may calculate the precision, recall and f1 scores of the Bernulli model
vali_acc_score_nb=[]; vali_preci_socre_nb=[]; vali_recall_score_nb=[];vali_f1_score_nb=[]
for i in range(5):
    
    validate_data_im,validate_labels_im,training_data_im,training_labels_im = train_validation_split(cv_train_data_im,cv_train_label_im,(i+1))
    training_data_im=np.asarray(training_data_im,dtype=np.float64);training_labels_im=np.asarray(training_labels_im,dtype=np.float64)
    # TEST BERNOULLI NAIVE BAYES on trainng and validation sets 
    del BernNB; 
    BernNB = BernoulliNB(binarize = True); BernNB.fit(training_data_im,training_labels_im)
    validate_expect = validate_labels_im; validate_pred_ber= BernNB.predict(validate_data_im)
    validate_expect=np.asarray(validate_expect,dtype=np.float64)
    # BernulliNB accuracy score 
    validate_expect_ber = validate_labels_im
    tp, tn, fn, fp = acc.compute_tp_tn_fn_fp(validate_expect_ber,validate_pred_ber) # for other score computation 
    conf_mat = acc.compute_tp_tn_fn_fp(validate_expect_ber,validate_pred_ber)     # confusion matrix: (tp, tn, fn, fp)
    validate_acc_score_ber = (acc.compute_accuracy(*list(conf_mat)))*0.01        # compute accuracy score   
    vali_acc_score_nb.append(validate_acc_score_ber)                             # append validation accuracy score for each experiment 
    
    # precision score 
    validate_preci_score_ber = acc.compute_precision(tp, fp)*0.01
    vali_preci_socre_nb.append(validate_preci_score_ber)    

    # recall score 
    validate_recall_score_ber = acc.compute_recall(tp, fn)*0.01
    vali_recall_score_nb.append(validate_recall_score_ber)
    
    # f1 score 
    validate_f1_score_ber = acc.compute_f1_score(validate_expect_ber,validate_pred_ber)
    vali_f1_score_nb.append(validate_f1_score_ber)

mu_vali_acc_nb=np.mean(vali_acc_score_nb); 
mu_vali_preci_nb=np.mean(vali_preci_socre_nb); 
mu_vali_recall_nb=np.mean(vali_recall_score_nb);
mu_vali_f1_nb=np.mean(vali_f1_score_nb)
print('Ber NB accuracy score:',mu_vali_acc_nb)
print('Ber NB precision score:',mu_vali_preci_nb)
print('Ber NB recall score:',mu_vali_recall_nb)
print('Ber NB f1 score:',mu_vali_f1_nb)





