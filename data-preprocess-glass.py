import numpy as np 
import urllib.request
import os

#put the data into an array
glass_data= np.genfromtxt('glass.data', delimiter=',')
print(glass_data)

#split into design matrix and labels