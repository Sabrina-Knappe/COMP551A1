import numpy as np 
import urllib.request
import os

#put the data into an array
glass_data= np.genfromtxt('glass.data', delimiter=',')
print(glass_data)

#split into design matrix and labels
glass_labels = glass_data[:, 10]
print(glass_labels)

glass_design_matrix= np.delete(glass_data, 10, 1)
print(glass_design_matrix)