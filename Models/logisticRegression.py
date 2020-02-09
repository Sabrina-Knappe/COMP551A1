#Implementing the logistic regression class
import numpy as np
import scipy

class Logistic_Regression:
  def __init__(self, step_size, name, class_type):
    if class_type == "binary":  #depends on the type of dataset we use
        self.type= True
    else: 
        self.type=False 
    self.name = name #string value
    self.learning_rate = 0.01
    self.it_num= 1000
    self.term = 1e-2 #termination condition
    self.regularization = 0 
    
    
def fit(self, training_data, training_labels, learning_rate, term):
    #Use gradient descent to generate best parameters (full batch)
    N,D = training_data.shape
    self.params= np.zeros(D)
    temp = np.inf

    while np.linalg.norm(temp) > term:
        if type == True: 
            temp = gradient(self,training_data, training_labels, self.params, self.regularization)
        else:
            onehot_labels = onehot(training_labels)
            temp = gradient(self,training_data, onehot_labels, self.params, self.regularization)
        self.params -= learning_rate*temp
    return self.params
    

def predict(self, params, test_data):
    N,D = test_data.shape

    if type==True: 
        y_pred = logistic(np.dot(test_data, params))
    else: 
        y_pred = softmax(results=np.dot(test_data,params))
    
    categories = np.argmax(y_pred,axis=0) # check if axis is right

    return categories
    #requires parameter values


def bin_cost(self, params, design_matrix, labels): 
    temp= np.dot(design_matrix, params)
    cost_func= np.mean(labels*np.log1p(np.exp(-temp))+(1-labels*np.log1p(np.exp(temp))))
    return cost_func

def multi_cost(self, params, design_matrix, labels): 
    temp= np.dot(design_matrix, params.T)
    cost = 0 - np.sum(np.dot(design_matrix, labels.T) - logsumexp(temp))
    return cost

def onehot(labels): 
    #one hot encoding
    #takes categorical data and puts it into matrices
    num_labels, num_classes = labels.shape[0], np.max(labels)
    onehot_labels = np.zeros(num_labels, num_classes)
    onehot_labels[np.arrange(num_labels), labels-1] = 1
    return onehot_labels

def softmax(results): 
    #performs softmax on an element
    y_pred = np.exp(results)
    y_pred /= np.sum(results)
    return y_pred

def logsumexp(vec): 
    #gives log of sum of exponents of elements of vec
    vec_max = np.max(vec,0)[None,:]
    result = vec_max + np.log(np.sum(np.exp(vec - vec_max)))
    return result

def gradient(self, design_matrix, labels, params, regularization): 
    #Finds gradient for a given set of params
    N,D = design_matrix.shape
    y_pred = logistic(np.dot(design_matrix, params)) 
    grad = np.dot(design_matrix.T, y_pred - labels)/N
    grad[1:] += regularization * params[1:] #L2 regularization
    # grad[1:] += regularization * np.sign(w[1:]) #L1 regularization
    return grad

def logistic(logit): 
    #Evaluates logistic function on logit
    fcn_value = 1/(1 + np.exp(-logit))
    return fcn_value
