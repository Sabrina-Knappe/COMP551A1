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
# df_names = pd.read_csv('adult_names.csv', sep=',')
# df_old_names = pd.read_csv('adult_old_names.csv', sep=',')

print(df_adult_train); print(df_adult_test) # see what the data is s

# Convert df_adult_data data frame to numpy
adult_train = df_adult_train.to_numpy()
adult_test = df_adult_test.to_numpy()

# Split array into design matrix and labels

# adult_train[1,:] a whole row 
# adult_train[:,1] a whole column 

# labels of training set alist[start:stop:step]
adult_train_label = adult_train[:,-1]
# labels of testing set 
adult_test_label = adult_test[:,-1]

# design matrix - the rest of the adult_train, adult_test excluding the last column 
train_desMat = adult_train[0:len(adult_train)-1:1]
test_desMat = adult_test[0:len(adult_train)-1:1]

# remove any examples with missing or malformed features in the training set 
# if the row has '?' at any column we delete the whole row 
# record the row number into row2del (row to delete)
row2del_train = []; 
for i in range(len(train_desMat)):
    row1=[]
    for row in enumerate(train_desMat[i,:]):
        row1.append(row[1]) 
    if ' ?' in row1:
        row2del_train.append(i)
# remove the rows indexed in row2del from the training set
train_desMat_new = train_desMat
train_desMat_new = np.delete(train_desMat,(row2del_train),axis=0) # 30161 training instances left 

# remove any examples with missing or malformed features in the testing set 
row2del_test = []; 
for i in range(len(test_desMat)):
    row1=[]
    for row in enumerate(test_desMat[i,:]):
        row1.append(row[1]) 
    if ' ?' in row1:
        row2del_test.append(i)
# remove the rows indexed in row2del from the training set
test_desMat_new = test_desMat
test_desMat_new = np.delete(test_desMat,(row2del_test),axis=0) # 15060 testing instances left 

# performing one-hot-encoding on categorical data and leave the numerical data be 
# (allowed to use sklearn forone hot encoding)
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def onehot_encode(data):

    # define example
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)


# more feature selection engineering before feeding into model 


        
        
        
        



