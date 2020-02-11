# -*- coding: utf-8 -*-
"""
Dataset pre-processing # 2: Adult Data 
Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. 
    A set of reasonably clean records was extracted using the following conditions: 
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.


"""

# Import package
# from numpy import array
# from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from urllib.request import urlretrieve

# Import pandas
import pandas as pd
# Import numpy
import numpy as np
import os 

# Assign url of file: url
url_index = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/Index'
url_data =  'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
url_names = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' 
url_old_names  ='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names'


# Save file locally
# data folder path 
dataFolderPath = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Dataset Folder\\Dataset2_Adult'
os.chdir (dataFolderPath)
# retreving the data
urlretrieve(url_index, 'adult_index.csv')
urlretrieve(url_data, 'adult_data.csv')
urlretrieve(url_names, 'adult_names.csv')
urlretrieve(url_test, 'adult_test.csv')
urlretrieve(url_old_names, 'adult_old_names.csv')


# Read file into a DataFrame and print its head, df = data frame 

df_index = pd.read_csv('adult_index.csv', sep=',', header=None) # no header, so first row can be read 
df_adult_train = pd.read_csv('adult_data.csv', sep=',', header=None) # adult data training set (including the validation set)
df_adult_test = pd.read_csv('adult_test.csv', sep=',', header=None)  # adult data testing set # somehow this only has one column...
# with the first row removed from the original testing dataset'|1x3 Cross validator']
print(df_adult_train); print(df_adult_test) # see what the data is s

# feature attributes - adding data frame headers using pandas 
df_adult_train.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df_adult_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
# concatenating the training and testing dataframes column wise; Stack the DataFrames on top of each other
# df_adult_all = pd.concat([df_adult_train, df_adult_test], axis=0)
df_adult_all = pd.concat([df_adult_train, df_adult_test])

# Convert df_adult_data data frame to numpy
adult_train = df_adult_train.to_numpy() # given
adult_test = df_adult_test.to_numpy() # given
adult_all = df_adult_all.to_numpy() # used for k-fold cross validation 

# Split array into design matrix and labels

# adult_train[1,:] a whole row 
# adult_train[:,1] a whole column 

# labels of training set alist[start:stop:step]
adult_train_label = adult_train[:,-1]
# labels of testing set 
adult_test_label = adult_test[:,-1]
# labels of all data 
adult_all_label = adult_all[:,-1]


# design matrix - the rest of the adult_train, adult_test and all data excluding the last column 
train_desMat = adult_train[:,0:len(adult_train[1,:])-1:1]
test_desMat = adult_test[:,0:len(adult_train[1,:])-1:1]
all_desMat = adult_all[:,0:len(adult_all[1,:])-1:1]

# remove any examples with missing or malformed features in the training set 
# if the row has '?' at any column we delete the whole row 
# record the row number into row2del (row to delete)
row2del_all = []; 
for i in range(len(adult_all)):
    row1=[]
    for row in enumerate(adult_all[i,:]):
        row1.append(row[1]) 
    if ' ?' in row1:
        row2del_all.append(i)
# remove the rows indexed in row2del from the training set
adult_all_new = adult_all
adult_all_new = np.delete(adult_all,(row2del_all),axis=0) # 45222 training instances left 
# drop these rows in the dataframe too by row number: df_adult_all

df_adult_all_new = df_adult_all.drop(df_adult_all.index[row2del_all])
# dataframe of processed instances (total=42960) and 15 features 
# change this total dataframe into numpy and obtain their labels 
adult_all_new_processed = df_adult_all_new.to_numpy()
# GET THE LABELS FOR ALL INSTANCES 
adult_all_new_labels = adult_all_new_processed[:,-1]

# performing one-hot-encoding on categorical data and leave the numerical data be 
# (allowed to use sklearn forone hot encoding)
fp = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Data Preprocessing'
os.chdir (fp)
from util_preprocess import onehot_encode
# process for df_adult_all, it works with dataframes, df_adult_all is unprocessed (with '?'), we remove row2del to all instances afterwards 
# producings feautres_all
data=df_adult_all
# ALL DATA TO BE SPLTTED INTO TRAIN-VALIDATION-TEST SETS 
# onehot_encode(data): 
# performs one-hot-encoding on categorical data and then concatenate back with the numerical columns 

    # Select the numeric columns in training set data frame:df_adult_train
numeric_subset = data.select_dtypes('number')
categorical_subset = data.select_dtypes('object') 
        # One hot encode
categorical_subset = pd.get_dummies(categorical_subset[categorical_subset.columns.drop('income')])
        # categorical_subset = pd.get_dummies(categorical_subset[categorical_subset.columns.drop(14)])
        
        # re-Join the categorical dataframe (one-hot-encoded) and the numeric dataframe using concat
        # Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)
print (features.head())   # printing the header out 
      
        # Replace the inf with nan
features = features.replace({np.inf: np.nan, -np.inf: np.nan})   
        # Drop na values
features = features.dropna()
        
    # return features, numeric_subset, categorical_subset


# features, numeric_subset, categorical_subset = onehot_encode(data_all)
features_all, numeric_subset_all, categorical_subset_all = features, numeric_subset, categorical_subset

features_all = features.drop(features.index[row2del_all])# 42960 instances left and 108 columns for features 
# including 6 numeric columns: age,fnlwgt,educational-num,capital-gai, capital-loss, hours-per-week
# the rest are categorical and are one-hot-encoded! 

# convert to numpy to be fed into models 
features_all_np = features_all.to_numpy()
# for naive bayes header 
#type of feature it is ("continuous", "binary", "categorical")
type_cont = ['continuous']*6 # numerical data 
type_bi = ['binary']*102 # one-hot-encoded categorical data 
feature_types = np.concatenate((type_cont, type_bi), axis=None)

labels_all_np = adult_all_new_labels

# To feed data into Naive Bayes, we need 
# features_all_np: a numpy array with 6 columns of numeric data, 102 columns of binary one-hot-encoded data, and 42960 instances 
# feature_types: continuous or binary for 108 columns
# labels_all_np: labels of all 42960 instances 

###############################################################################
# USELESS CODE FROM HERE ON
###############################################################################
# remove any examples with missing or malformed features in the training set 
# if the row has '?' at any column we delete the whole row 
# record the row number into row2del (row to delete)
row2del_train = []; 
for i in range(len(adult_train)):
    row1=[]
    for row in enumerate(adult_train[i,:]):
        row1.append(row[1]) 
    if ' ?' in row1:
        row2del_train.append(i)
# remove the rows indexed in row2del from the training set
adult_train_new = adult_train
adult_train_new = np.delete(adult_train,(row2del_train),axis=0) # 30161 training instances left 

# remove any examples with missing or malformed features in the testing set 
row2del_test = []; 
for i in range(len(adult_test)):
    row1=[]
    for row in enumerate(adult_test[i,:]):
        row1.append(row[1]) 
    if ' ?' in row1:
        row2del_test.append(i)
# remove the rows indexed in row2del from the training set
adult_test_new = adult_test
adult_test_new = np.delete(adult_test,(row2del_test),axis=0) # 15060 testing instances left 

# concatenating the designmatrix of the processed training and testing sets 
# concatenate 2 numpy arrays: row-wise
# adult_all_new = np.concatenate((adult_train_new, adult_test_new)) 

# convert adult_all_new back into data frames using pandas and using one hot encoding with the "income" column at last 
df_adult_all_new =  pd.DataFrame(data=adult_all_new, columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
# now all data has size of 45222 x 15 

# performing one-hot-encoding on categorical data and leave the numerical data be 
# (allowed to use sklearn forone hot encoding)
fp = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Data Preprocessing'
os.chdir (fp)
from util_preprocess import onehot_encode
# process for df_adult_all, it works with dataframes 
data_all = df_adult_all_new # run inside util_preprocess.py; data = data_all; 45222 by 15 (including "income")

# ALL DATA TO BE SPLTTED INTO TRAIN-VALIDATION-TEST SETS 
features, numeric_subset, categorical_subset = onehot_encode(data_all)
features_all, numeric_subset_all, categorical_subset_all = features, numeric_subset, categorical_subset
# features_all is numerical (4) + OHE categorical (102) feautres for all processed 48842 instances 
# numeric_subset_all: numerical (4) features, 48842 instances 
# categorical_subset_all: categorical (102) features, 48842 instances 

# ALL LABELS FOR TRAIN-VALIDATION-TEST SETS 


###############################################################################
# 5-fold cross validaiotn 
# splitting of the dataset 


###############################################################################
# basic statistics of train_desMat and test_desMat 
os.chdir (fp)
from util_preprocess import basic_stats
basic_stats(df_adult_train)
###############################################################################

# more feature selection engineering before feeding into model 
    
    


#################################################################################################3
# create dictionary to know column name features_dict[1] 

# select the onehot_encode function from util_oreoricess
# to excute the code equivalent to onehot_encode(data) obtaining features 
# features_train = features; "education_nums" already deleted
features_train, numeric_subset_train, categorical_subset_train = onehot_encode(data_train)
# resulting in 107 columns after one-hot-encoding 
# no need to do one-hot encoding for testing dataset 
#################################################################################################



        
        
        
        



