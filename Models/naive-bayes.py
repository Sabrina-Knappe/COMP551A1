#Implementing the naive bayes class
class Naive_Bayes(object):
    feature_types=[]
    prior_type=""
    name=""
    
    def __init__(self, step_size, name, feature_types, prior_type):
        self.name = name
        self.prior_type= prior_type

    

def fit(self, training_data, training_labels):
    model= 0
    prior= 0
    print("Hello my name is " + self.name)
    #learn the prior probabilities
    if(self.prior_type=="multiclass"):
        prior=self.multiclass(training_data, training_labels)
    else:
        prior=self.binary(training_data, training_labels)
    #learn the likelihood components
    if()
    return model


#class prior
def multiclass(self, training_data, training_labels):
    #use the multi version of max likelihood??
    return 0


def binary(self, training_data, training_labels):
    #use the max likelihood estimate
    return 0


#likelihood


def predict(self, test_data, test_labels):
    category= 0
    print("We predict that "+self.name+"has label "+category)
    return category
