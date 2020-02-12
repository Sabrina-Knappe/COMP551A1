#Implementing the naive bayes class
import numpy as np

class Naive_Bayes(object):
    feature_types=np.array([])
    name=""


    #creating the Naive Bayes model for your dataset.
    #name is a string that is the name of the dataset
    #feature_types is an array of strings that is analogous with the positions of the features for your data and tells what
    #type of feature it is ("continuous", "binary", "categorical")
    def __init__(self, name, feature_types):
        self.name = name
        self.feature_types= feature_types

    

    def fit(self, training_data, training_labels, test_data):
        model= 0
        prior= 0
        print("Now fitting " + self.name)
        #partition the data according to the types of the
        categories, inverse =np.unique(training_labels, False, True, False)
        j=0
        # print(training_labels.shape)
        hot_labels= self.onehot(training_labels)
        tupe= (1, training_labels.shape[0])
        binary= np.array([])
        categorical= np.array([])
        continuous= np.array([])
        binary_model= np.array([])
        categorical_model= np.array([])
        continuous_model= np.array([])
        for c in inverse:
            if(categories[c]=="binary"):
                print(training_data[:, j])
                np.append(binary, training_data[:, j], axis=0)
            elif(categories[c]=="categorical"):
                print(training_data[:, j])
                np.append(categorical, training_data[:, j], axis=0)
            else:
                # print("continuous")
                # print(continuous.ndim)
                # a = np.array([[1, 2], [3, 4]])
                if(continuous.ndim==1):
                    continuous= np.array([training_data[:, j]])
                    continuous=continuous.T
                else:
                    b = np.array([training_data[:, j]])
                    continuous= np.concatenate((continuous, b.T), axis=1)
                # print(continuous)
            # print("HELLO")
            # print(training_data.shape[1])
            if(j<(training_data.shape[1]-1)):
                j=j+1
        
        if(binary.shape[0]!=0):
            # print("hit")
            binary_model= self.binary_likelihood(binary, hot_labels)
        if(categorical.shape[0]!=0):
            # print("hit")
            categorical_model= self.categorical(categorical, hot_labels)
        if(continuous.shape[0]!=0):
            # print("hit")
            # print(continuous)
            continuous_model= self.continuous(continuous, hot_labels, test_data)
        model= np.sum([binary_model, categorical_model, continuous_model])
        return model


    def predict(self, test_data, test_labels, params):
        predictions= np.array()
        for t in test_data:
            options= np.multiply(params, t)
            prediction= np.argmax(options)
            print("We predict that "+t+"has label "+prediction)
            np.append(predictions, prediction)
        return predictions

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
        count_sample = training_data.shape[0]
        separated = [[x for x, t in zip(training_data, training_labels) if t == c] for c in np.unique(training_labels)]
        prior = [np.log(len(i) / count_sample) for i in separated]
        print("prior ")
        print(prior)
        count = np.array([np.array(i).sum(axis=0) for i in separated])
        n_doc = np.array([len(i) for i in separated])
        likelihood = count / n_doc[np.newaxis].T
        print("likelihood ")
        print(likelihood)
        return prior+likelihood

    def categorical(self, training_data, training_labels):
        count_sample = training_data.shape[0]
        separated = [[x for x, t in zip(training_data, training_labels) if t == c] for c in np.unique(training_labels)]
        log_prior = [np.log(len(i) / count_sample) for i in separated]
        print("log prior "+log_prior)
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)
        print("likelihood "+feature_log_prob)
        return log_prior+feature_log_prob

    def continuous(self, training_data, training_labels, test_data):
        N= training_labels.shape[0]
        unique_labels=np.unique(training_labels)
        print(unique_labels)
        C= unique_labels.size
        print("HELP")
        print(C)
        D= training_data.shape[1]
        mu, s= np.zeros((C,D)), np.zeros((C,D))
        for c in range(C): #calculate mean and standard deviation
            inds=np.nonzero(training_labels[:,c])[0]
            mu[c,:]=np.mean(training_data[inds,:],0)
        log_prior= np.log(np.mean(training_labels,0))[:,None]
        print("log prior")
        print(log_prior)
        log_likelihood= - np.sum(.5*(((test_data[None, :, :] - mu[:,None,:]))**2), 2)  
        print("likelihood ")  
        print(log_likelihood)
        return log_prior+log_likelihood

    def onehot(self, labels): 
        #one hot encoding
        #takes categorical data and puts it into matrices
        categories = np.unique(labels)
        char_to_int = dict((c, i) for i, c in enumerate(categories))
        int_labels = [char_to_int[categories] for categories in labels]
        print(int_labels)

        # then make a matrix
        #num_labels, num_classes = labels.shape[0], np.max(int_labels)
        #onehot_labels = np.zeros(num_labels, num_classes)
        #onehot_labels[np.arange(num_labels), int_labels-1] = 1

        onehot_encoded = list()
        for value in int_labels:
            letter = [0 for _ in range(len(categories))]
            letter[value] = 1
            onehot_encoded.append(letter)
        print(onehot_encoded)
        onehot_encoded= np.array([onehot_encoded])

        return onehot_encoded