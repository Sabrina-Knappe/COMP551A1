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
    self.term = null #termination condition
    self.regularization = 0 
    
    
def fit(self, training_data, training_labels, learning_rate, term):
    #Use gradient descent to generate best parameters (full batch)
    N,D = training_data.shape
    self.params= np.zeros(D)
    temp = np.inf

    while np.linalg.norm(temp) > term:
        if(type == "binary")
            temp = gradient(training_data, training_labels, params)
        else:
            onehot_labels = onehot(training_labels)
            temp = gradient(training_data, onehot_labels, params)
        params = params - learning_rate*temp
    return params
    

def predict(self, params, test_data):
    N,D = test_data.shape

    if(type=="binary") 
        y_pred = logistic(np.dot(test_data, params))
    else
        y_pred = softmax(np.dot(test_data,params))
    
    categories = np.argmax(y_pred,axis=0) # check if axis is right

    category= 0
    print("We predict that "+self.name+"is category "+category)
    return category
    #requires parameter values


def binary_cost(self, params, design_matrix, labels)
    temp= np.dot(design_matrix, params)
    cost_func= np.mean(label*np.log1p(np.exp(-temp))+(1-labels*np.log1p(np.exp(temp))))
    return cost_func

def multi_cost(self, params, design_matrix, labels)
    temp= np.dot(design_matrix, params.T)
    cost = - np.sum(np.dot(design_matrix, labels.T) - logsumexp(temp))
    return cost

def onehot(self, labels)
    #one hot encoding
    #takes categorical data and puts it into matrices
    num_labels, num_classes = labels.shape[0], np.max(labels)
    onehot_labels = np.zeros(num_labels, num_classes)
    onehot_labels[np.arrange(num_labels), y-1] = 1
    return onehot_labels

def softmax(self,results)
    #performs softmax on an element
    y_pred = np.exp(results)
    y_pred /= np.sum(results)
    return y_pred

def logsumexp(vec)
    #gives log of sum of exponents of elements of vec
    vec_max = np.max(vec,0)[None,:]
    result = vec_max + np.log(np.sum(np.exp(vec - vec_max)))
    return result

def gradient(self, design_matrix, labels, params, regularization)
    #Finds gradient for a given set of params
    N,D = design_matrix.shape
    y_pred = logistic(np.dot(design_matrix, params)) 
    grad = np.dot(design_matrix.T, y_pred - y)/N
    grad[1:] += regularization * w[1:] #L2 regularization
    # grad[1:] += regularization * np.sign(w[1:]) #L1 regularization
    return grad

def logistic(logit)
    #Evaluates logistic function on logit
    fcn_value = 1/(1 + np.exp(-logit))
    return fcn_value