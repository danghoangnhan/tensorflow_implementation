import numpy as np
from numpy import log, dot, e,int16
from numpy.random import rand
from sklearn.preprocessing import LabelEncoder

"""
Created on Fri April  11 04:02:05 2023
@author: danghoangnhan.1@gmail.com
"""


# Linear Regression
class LinearRegression:

    def __init__(self, lr=0.001, weights=None, bias=0):
        self.learningrate = lr
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = rand(n_features).reshape(-1, 1)
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_val = labelencoder.fit_transform(y_val)
        
        for i in range(1, self.iteration):
            y_pred = self.predict(X)
            gradient = np.dot(X.T, (y_pred - y.reshape(-1, 1))) / n_samples
            self.weights -= self.learning_rate * gradient
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Logistic Regression
class LogisticRegression:
    def __init__(self,lr=0.001):
        self.lr = lr
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + e ** (-z))

    def cost_function(self, X, y, weights):
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)

    def fit(self, X_train, y_train,x_val,y_val, epochs):
        n_samples, n_features = X_train.shape
        self.weights = rand(n_features).reshape(-1, 1)
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_val = labelencoder.fit_transform(y_val)
        for _ in range(epochs):
            z = np.dot(X_train, self.weights)
            y_pred = self.sigmoid(z)            
            gradient = np.dot(X_train.T, (y_pred - y_train.reshape(-1, 1))) / n_samples
            gradient_bias = np.mean(y_pred - y_train.reshape(-1, 1))
            self.weights -= self.lr * gradient
            self.bias -= self.lr * gradient_bias

            predictions = self.predict(x_val)
            correct_predictions = np.sum(predictions == y_val)
            accuracy = correct_predictions / len(y_val)
            print("accuracy:",accuracy)  

    def predict(self, X):
        z = dot(X, self.weights) + self.bias
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

    def Out(self, X):
        y_predict = self.predict(X)
        return [[self.labelDecodeMapping[value]] for value in y_predict]

    def Encode(self, Y):
        result = []
        count = 0
        for value in Y:
            if value[0] not in self.labelEncodeMapping:
                self.labelEncodeMapping[value[0]] = count
                self.labelDecodeMapping[count] = value[0]
                count += 1
            result.append([self.labelEncodeMapping[value[0]]])
        return np.asarray(result)
