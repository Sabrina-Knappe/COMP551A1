# -*- coding: utf-8 -*-
"""
Dataset pre-processing # 2: Adult Data 
Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. 
    A set of reasonably clean records was extracted using the following conditions: 
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.


- Yujing Zou
"""

# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd
# Import numpy
import numpy as np

# Assign url of file: url
url_data =  'https://archive.ics.uci.edu/ml/datasets/Adult/adult.data'

# Save file locally
urlretrieve(url, 'adult-data.csv')

# Read file into a DataFrame and print its head
df = pd.read_csv('adult-data.csv', sep=',', error_bad_lines=False)
x = np.genfromtxt("adult-data.csv", dtype=None)
print(df)




# Convert to numpy
temp = df.to_numpy()

# Split array into design matrix and labels


