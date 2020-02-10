# -*- coding: utf-8 -*-

"""
    In addtion to model classes, 
    
    define a function evaluate_acc to evaluate the model accuracy
        
        This function takes the true labels (i.e. y) and the target labels (i.e. y_hat) as input
        and it should output the accuracy score
        
    define a function to run k-fold cross validation 
"""

def evaluate_acc(y,y_hat):
    '''
    input: 
        y: true labels  
        y_hat: target labels 
        
    output:
        accuracy score of the models 
    '''
# Accuracy, Recall, Precision, F1 Score in Python from scratch
    # https://www.youtube.com/watch?v=9PbrWiLC-4k
    
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae



### classification metrics - Accuracy, Recall, Precision, F1 Score
# https://github.com/bhattbhavesh91/classification-metrics-python/blob/master/ml_a.ipynb
    
