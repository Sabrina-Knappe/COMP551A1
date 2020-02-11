#Implementing the naive bayes class
class Naive_Bayes(object):
    feature_types=[]
    prior_type=""
    name=""
    
    #creating the Naive Bayes model for your dataset.
    #name is a string that is the name of the dataset
    #feature_types is an array of strings that is analogous with the positions of the features for your data and tells what
    #type of feature it is ("continuous", "binary", "categorical")
    def __init__(self, name, feature_types):
        self.name = name
        self.prior_type= prior_type
        self.feature_types= feature_types

    

def fit(self, training_data, training_labels):
    model= 0
    prior= 0
    print("Now fitting " + self.name)
    #partition the data according to the types of the
    categories, inverse =np.unique(training_labels, False, True, False)
    j=0
    binary= np.array()
    categorical= np.array()
    continuous= np.array()
    for c in inverse:
        if(categories[c]=="binary"):
            np.append(binary, training_data[:, j])
        elif(categories[c]=="categorical"):
            np.append(categorical, training_data[:, j])
        else:
            np.append(continuous, training_data[:, j])
        j++
    binary_model= self.binary_likelihood(training_data, training_labels)
    categorical_model= self.categorical(training_data, training_labels)
    continuous_model= self.continuous(training_data, training_labels)
    model= np.sum(binary_model,categorical_model,continuous_model)
    return model


def predict(self, test_data, test_labels):
    category= 0
    print("We predict that "+self.name+"has label "+category)
    for c in class:
        #multiply array[c_index]*
    return category

#class prior
def multiclass(self, training_data, training_labels):
    #use the multi version of max likelihood??
    #Count number of occurrences of each value in array of non-negative ints.
    categories, number_of_each= np.unique(training_labels, False, False, True)
    N= training_labels.size
    i=0
    max_likelihood_estimate= []
    for c in categories:
        max_likelihood_estimate[i]= number_of_each[i]/N
    return max_likelihood_estimate


def binary_prior(self, training_data, training_labels): #do i need to change this?
    categories, number_of_each= np.unique(training_labels, False, False, True)
    N= training_labels.size
    i=0
    max_likelihood_estimate= []
    for c in categories:
        max_likelihood_estimate[i]= number_of_each[i]/N
    return max_likelihood_estimate


#likelihood
def binary_likelihood(self, training_data, training_labels):
    #implement bernouilli naive bayes
    prior= self.multiclass(training_data, training_labels)
    #likelihood of each word under each class
    likelihood= self
    logp= np.log(prior)+np.sum(np.log(likelihood*x[:,None]),0)+np.sum(np.log((1-likelihood)*(1-x[:,None])),0)
    posterior= np.exp(log_p0)
    posterior /= np.sum(posterior)
    return posterior

def categorical(self, training_data, training_labels):
    return 0

def continuous(self, training_data, training_labels):
    N, C= training_labels.shape
    D= training_data.shape[1]
    mu, s= np.zeros((C,D)), np.zeros((C,D))
    for c in range(C): #calculate mean and standard deviation
        inds=np.nonzero(training_labels[:,c])[0]
        mu[c,:]=np.mean(training_data[inds,:],0)
    log_prior= np.log(np.mean(y,0))[:,None]
    log_likelihood= - np.sum(.5*(((Xt[None, :, :] - mu[:,None,:]))**2), 2)    
    return log_prior+log_likelihood

