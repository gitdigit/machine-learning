# we will implement a k-fold cross validation from scratch
# we will use the iris dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
#the number of folds
k = 5

#3 - faire un loop for que nous allons itérer k fois. Avant le loop for, il faut définir les 4 variables x_train, y_train, x_test, y_test sachant qu'on doit définir k fois les 4 à chaque fois
def k_fold_cross_validation(X, y, k, model):
    # X is the data
    # y is the target
    # k is the number of folds - quantité de morceau qu'on va diviser
    # model is the model to use  
    # we will return the accuracy of the model
    # we will use the accuracy as a metric

    #################################
    # shuffle the data and create X and y ready to be used to fit the model
    # in a way that if I say X[0] the algorithm will return the first fold  of the data, the same for y
    #################################
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    kFold = StratifiedKFold(n_splits=k)
    
    # we will need to define a for loop to iterate over the folds and guarantee that each fold is used as a test set at least once
    # inside this for loop we will call the functions fit and accuracy for each one of the folds
    # X_train, y_train, X_test, y_test are build each time the for loop is called by using X and y divided before
    
    accuracies = []
    #######Your for loop here

    # Your code to define X_train, y_train, X_test, y_test
    for train_index, test_index in kFold.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    # Logistic Regression as an example model
        model = LogisticRegression()
    # we will fit the model on the train data - enlever le commentaire après, c'était pour ne pas pertuber la fonction
        model.fit(X_train, y_train)
     # we will predict on test
        y_pred = model.predict(X_test)
    # we will compute the accuracy
        accuracy = np.mean(y_pred == y_test)
    # we will append the accuracy to the list
        accuracies.append(accuracy)
    
    # we will return the mean accuracy
    return np.mean(accuracies)
#the model we are using
model = LogisticRegression(max_iter=200)
mean_accuracy = k_fold_cross_validation(X, y, k, model)
print(f"Mean accuracy over 5 folds: {mean_accuracy:.4f}")
