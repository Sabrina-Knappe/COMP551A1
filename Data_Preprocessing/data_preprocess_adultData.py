# -*- coding: utf-8 -*-
"""
Dataset pre-processing # 2: Adult Data 
Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. 
    A set of reasonably clean records was extracted using the following conditions: 
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.


"""
def preprocess_adult():
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
    # dataFolderPath = '../Dataset_Folder/Dataset2_Adult/'
    # os.chdir (dataFolderPath)
    # # retreving the data
    # urlretrieve(url_index, 'adult_index.csv')
    # urlretrieve(url_data, 'adult_data.csv')
    # urlretrieve(url_names, 'adult_names.csv')
    # urlretrieve(url_test, 'adult_test.csv')
    # urlretrieve(url_old_names, 'adult_old_names.csv')
    
    
    # Read file into a DataFrame and print its head, df = data frame 
    
    df_index = pd.read_csv('Dataset_Folder/Dataset2_Adult/adult_index.csv', sep=',', header=None) # no header, so first row can be read 
    df_adult_train = pd.read_csv('Dataset_Folder/Dataset2_Adult/adult_data.csv', sep=',', header=None) # adult data training set (including the validation set)
    df_adult_test = pd.read_csv('Dataset_Folder/Dataset2_Adult/adult_test.csv', sep=',', header=None)  # adult data testing set # somehow this only has one column...
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
    fp = 'Dataset_Folder/Dataset2_Adult/adult_data.csv'
    os.chdir (fp)
    # from util_preprocess import onehot_encode
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
    type_bi = ['binary']*(108 - 6) # one-hot-encoded categorical data 
    feature_types = np.concatenate((type_cont, type_bi), axis=None)
    feature_types = np.transpose (feature_types) 
    
    labels_all_np = adult_all_new_labels # total of 42960 labels corresponding to features_all_np
    
    # now go into kfold_cross_validation.py to split training and testing set then undergo kfold CV
    
    # features_all_np: a numpy array with 6 columns of numeric data, 102 columns of binary one-hot-encoded data, and 42960 instances 
    # feature_types: continuous or binary for 108 columns
    # labels_all_np: labels of all 42960 instances 
    
    ###############################################################################
    #basic statistics of train_desMat and test_desMat 
    os.chdir (fp)
    from Data_Preprocessing.util_preprocess import basic_stats
    basic_stats(df_adult_train)
    
    return features_all_np, feature_types, labels_all_np
    
    
    
            
            
            
            
    
    
    
