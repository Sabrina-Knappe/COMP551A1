# -*- coding: utf-8 -*-
"""
Data preprocessing util functions 
"""
# from urllib.request import urlretrieve

# Import pandas
import pandas as pd
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# import sklearn.cross_validation as cross_validation
import sklearn.model_selection as model_selection
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# data_train = df_adult_train
# data = data_train
# select below to excute the code equivalent to onehot_encode(data) obtaining features 

# df = df_adult_train 
def basic_stats(df):
    for column in df.columns:
        print (column)
        if df.dtypes[column] == np.object: # Categorical data
            print (df[column].value_counts())
        else:
            print (df[column].describe())         
        print ('\n')
       
        # some fancy histograms 
    
# basic_stats(df_adult_train )

# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


# Calculate the correlation and plot it
# encoded_data, _ = number_encode_features(df)
    # show heatmap - exploring correlations 
# encoded_data = result 
# sns.heatmap(encoded_data.corr(), square=True)
# plt.show()
# encoded_data.tail(5)
# Expore the strong correaltion between "education" and "education-num" revealed by the heatmap (lighter color)
# df[["education", "educational-num"]].head(10)
# "education" and "education-num" are essentially the same data, delete the numerical one
# del df["educational-num"]
# df.head(1) to see "educational-num" is successfully deleted 
# also seems like "gender" and "relationship" are anti-correlation (darkest color on heatmap)


# data_train = df_adult_train
# data = data_train
# select below to excute the code equivalent to onehot_encode(data) obtaining features 


def onehot_encode(data):
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
    
    return features, numeric_subset, categorical_subset



  