import numpy as np
from numpy import log, dot, e,int16
from numpy.random import rand

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

        cost = []
        for i in range(1, self.iteration):
            r = self.predict(X) - y
            cost.append(.5 * np.sum(r * r))
            # self.weights[0] -= self.learningrate * np.sum(r)
            # # correct the shape dimension
            # self.weights[1] -= self.learningrate * np.sum(np.multiply(r, X[:, 1].reshape(-1, 1)))
            self.weights -= self.learningrate * np.sum(r) / n_samples
        self.loss = cost
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Logistic Regression
class LogisticRegression:
    def __init__(self,lr=0.001):
        self.labelEncodeMapping = {}
        self.labelDecodeMapping = {}
        self.lr = lr
        self.weights = None
        self.loss = None

    def sigmoid(self, z):
        return 1 / (1 + e ** (-z))

    def cost_function(self, X, y, weights):
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)

    def fit(self, X_train, y_train,x_val,y_val, epochs):
        n_samples, n_features = X_train.shape
        self.loss = []
        self.weights = rand(n_features).reshape(-1, 1)
        y_train = self.Encode(y_train)
        for _ in range(epochs):
            # Gradient Descent
            prediction = self.sigmoid(dot(X_train, self.weights))
            self.weights -= self.lr * dot(X_train.T, prediction - y_train) / n_samples
            # Saving Progress
            self.loss.append(self.cost_function(X_train, y_train, self.weights))

    def predict(self, X):
        z = dot(X, self.weights)
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
