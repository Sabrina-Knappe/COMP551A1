# -*- coding: utf-8 -*-
"""
SPECTF Heart Data Set

Abstract: Data on cardiac Single Proton Emission Computed Tomography (SPECT) images. 
Each patient classified into two categories: normal and abnormal.


"""


# Import package
from urllib.request import urlretrieve
# Import pandas
import pandas as pd
# Import numpy
import numpy as np
import os 

# Assign url of file: url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/'
url_SPECT_names = url + 'SPECT.names'
url_SPECT_test = url + 'SPECT.test'
url_SPECT_train = url + 'SPECT.train'
url_SPECTF_names = url + 'SPECTF.names'
url_SPECTF_test = url + 'SPECTF.test'
url_SPECTF_train = url + 'SPECTF.train'


# Save file locally
# data folder path 
dataFolderPath4 = 'D:\\Documents\\U4 Academics\\Winter 2020\\COMP 551-Applied Machine Learning\\Assignments\\Assignment 1\\COMP551A1\\Dataset Folder\\Dataset4_SPECTF_Heart'
os.chdir (dataFolderPath4)

# retreving the data
urlretrieve(url_SPECT_names, 'SPECT_names.csv')
urlretrieve(url_SPECT_test, 'SPECT_test.csv')
urlretrieve(url_SPECT_train, 'SPECT_train.csv')
urlretrieve(url_SPECTF_names, 'SPECTF_names.csv')
urlretrieve(url_SPECTF_test, 'SPECTF_test.csv')
urlretrieve(url_SPECTF_train, 'SPECTF_train.csv')

# Read file into a DataFrame and print its head, df = data frame 

df_wdbc_train_data = pd.read_csv('wdbc_data.csv', sep=',')
df_wpbc_test_data = pd.read_csv('wpbc_data.csv', sep=',')
# df_wdbc_train_names = pd.read_csv('wdbc_names.csv', sep=',') # adult data training set (including the validation set)
# df_wpbc_test_names = pd.read_csv('wpbc_names.csv', sep=',') # adult data training set (including the validation set)

# df_names = pd.read_csv('adult_names.csv', sep=',')
# df_old_names = pd.read_csv('adult_old_names.csv', sep=',')

print(df_wdbc_train_data) # see what the data is s

# Convert df_adult_data data frame to numpy
adult_train = df_wdbc_train_data.to_numpy()
adult_test = df_wdbc_train_data.to_numpy()


# Split array into design matrix and labels

# labels of training set 
train_label = adult_train[:,-1]
# labels of testing set 
test_label = adult_test[:,-1]

# design matrix 
