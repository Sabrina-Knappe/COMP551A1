#Implementing the naive bayes class
class Naive_Bayes(object):
    feature_types=[]
    prior_type=""
    name=""
    
    #creating the Naive Bayes model for your dataset.
    #name is a string that is the name of the dataset
    #feature_types is an array of strings that is analogous with the positions of the features for your data and tells what
    #type of feature it is (continuous, binary, categorical)
    #prior type
    def __init__(self, step_size, name, feature_types, prior_type):
        self.name = name
        self.prior_type= prior_type
        self.feature_types= feature_types

    

def fit(self, training_data, training_labels):
    model= 0
    prior= 0
    print("Now fitting " + self.name)
    #learn the prior probabilities
    if(self.prior_type=="multiclass"):
        prior=self.multiclass(training_data, training_labels)
    else:
        prior=self.binary_prior(training_data, training_labels)
    i=0
    for t in training_data:
        if(self.feature_types[i]=="binary"):
            self.binary_likelihood()
        elif(self.feature_types[i]=="categorical"):
            self.categorical()
        else:
            self.continuous()

    return model


def predict(self, test_data, test_labels):
    category= 0
    print("We predict that "+self.name+"has label "+category)
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
    
    return 0

def categorical(self, training_data, training_labels):
    return 0

def continuous(self, training_data, training_labels):
    return 0