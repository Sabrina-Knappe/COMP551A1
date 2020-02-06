#Implementing the logistic regression class
import numpy as np
import scipy

class Logistic_Regression:
  def __init__(step_size, name, type):
    if(type=="binary")  #depends on the type of dataset we use
        self.type= True
        self.cost= binary_cost()
    else
        self.type=False 
        self.cost= multi_cost()
    self.name = name #string value
    self.learning_rate = null
    self.it_num= null
    
    
def fit(self, training_data, training_labels):
    #Use gradient descent to generate best parameters (full batch)
    print("Hello my name is " + self.name)
    self.params= np.array()
    #cost function here?????

def predict(self, test_data, test_labels):
    category= 0
    print("We predict that "+self.name+"is category "+category)
    return category
    #requires parameter values


def binary_cost(self, params, design_matrix, labels,)
    temp= np.dot(design_matrix, params)
    cost_func= np.mean(label*np.log1p(np.exp(-temp))+(1-labels*np.log1p(np.exp(temp))))
    return cost_func

def multi_cost(self, params, design_matrix, labels,)
    temp= np.dot(design_matrix, params)
    cost_func= np.mean(label*np.log1p(np.exp(-temp))+(1-labels*np.log1p(np.exp(temp))))
    return cost_func