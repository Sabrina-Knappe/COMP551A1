# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:56:03 2020

Dataset pre-processing 
- Yujing 
"""

# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd

# Assign url of file: url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'

# Save file locally
urlretrieve(url, 'ionosphere-data.csv')

# Read file into a DataFrame and print its head
df = pd.read_csv('ionosphere-data.csv', sep=',')
print(df)

