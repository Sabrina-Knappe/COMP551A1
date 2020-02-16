# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
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


# %%
# Import numpy
import numpy as np

# Convert to numpy
#df.to_numpy()
temp = df.values
R,C = temp.shape
#print(C)

# Split array into design matrix and labels
ionosphere_labels = temp[:, C-1]
print(ionosphere_labels)


# %%
# Remove labels to get design matrix
ionosphere_design_matrix= np.delete(temp, C-1, 1)
print(ionosphere_design_matrix)


# %%
import Models.kfold_CV_try as cv

# Train test split
xTrain_ion, xTest_ion, yTrain_ion, yTest_ion = cv.split_train_test(ionosphere_design_matrix, ionosphere_labels, 0.2)
# print(xTest_ion); print(xTrain_ion); print(yTest_ion);print(yTrain_ion)


# %%
folds = 5 # delete folds later when embedded in function input
# cv_train_data_ion is xTrain_ion split into five chunks
dataset_split_in, cv_train_data_ion,cv_train_label_ion= cv.kfold_cross_validation(xTrain_ion,yTrain_ion,folds)
# print(cv_train_data_ion,cv_train_label_ion)

# the last input for cv.train_validation_split is the number of experiments you are running currently, 5 in total. 
# each experiments is organizing the five chunks from cv_train_data_ion into 4 chunks for training_data_ion and 1 chunk for validate_data_ion for cross validation, just need to uncomment the line you want to experiment currently. 

#exp1: 
validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = cv.train_validation_split(cv_train_data_ion,cv_train_label_ion,1)

#exp2:
#validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = cv.train_validation_split(cv_train_data_ion,cv_train_label_ion,2)

#exp3:
#validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = cv.train_validation_split(cv_train_data_ion,cv_train_label_ion,3)

#exp4:
#validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = cv.train_validation_split(cv_train_data_ion,cv_train_label_ion,4)

#exp5: 
#validate_data_ion,validate_labels_ion,training_data_ion,training_labels_ion = cv.train_validation_split(cv_train_data_ion,cv_train_label_ion,5)
print(validate_data_ion,validate_labels_ion)
print(validate_data_ion.shape,validate_labels_ion.shape)


# %% Obtaining parameters of LOGISTIC REGRESSION FOR IONOSPHERE; LEARNING RATE = 0.02
import Models.logisticRegression as lr

ionslr1 = lr.Logistic_Regression(0.02,"Ionosphere","binary") # input step size
# params = ionslr1.fit(cv_train_data, cv_train_label, 0.02, 1e-1)
params1 = ionslr1.fit(training_data_ion,training_labels_ion,0.02,1e-1)
#adding a separate comment
# input learning rate and termination conditions

# %% (Q3 part 2) EXPLIRING DIFFERENT LEARNING RATES AS A FUNCTION OF THE ACCURACY SCORE 
ionslr2 = lr.Logistic_Regression(0.05,"Ionosphere","binary") # input step size
params2 = ionslr1.fit(training_data_ion,training_labels_ion,0.005,1e-1)




# %% investigating the converging rate of the parameters as we change the learning rate! 
import matplotlib.pyplot as plt 
xaxis=list(range(len(params1))); print(xaxis)
plt.figure()
plt.plot(xaxis,params1,label='learning rate=0.02,term=1e-1'); plt.legend(loc="upper right")
plt.plot(xaxis,params2,label='learning rate=0.005,term=1e-2'); plt.legend(loc="upper right")
plt.xlabel('iterations'); plt.ylabel('LR Params'); plt.title('LR Converging rate');#plt.legend('0.02')
#plt.plot(); # overlay on the same figure
plt.show()
#legend()
#legend(labels)
# %%
# Making LR prediction on the validatoin set
print(validate_data_ion,validate_labels_ion)

predictions1 = lr.predict(ionslr1,params1,validate_data_ion)


# %%


