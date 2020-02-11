import numpy as np 
import urllib.request
import os

# def preprocess():
    #put the data into an array

def preprocess():
    glass_data= np.genfromtxt('Dataset_Folder/glass.data', delimiter=',')
    print(glass_data)

        #split into design matrix and labels
    glass_labels = glass_data[:, 10]
    print(glass_labels)

    glass_design_matrix= np.delete(glass_data, 10, 1)
    print(glass_design_matrix)

    return glass_design_matrix, glass_labels