from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import genfromtxt

traningData = genfromtxt('train.csv', delimiter=',', skip_header=1, dtype=np.float128)
testData = genfromtxt('test.csv', delimiter=',', skip_header=1, dtype=np.float128)

eta = .05
iterations = 2
learning_rate = 0.0015
x = np.delete(traningData, 0, axis=1)
y = np.reshape([0 if index == 2 else 1 for index in traningData[:, 0]], (x.shape[0], 1))


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def predict(index):
    return 5 if index >= 0.5 else 2


def logicstic_regisstion(X, Y, learning_rate, iterations):
    m = X.shape[0]
    n = X.shape[1]

    W = np.array([0.5 for _ in range(x.shape[0])]).reshape(m, 1)
    B = 1
    cost = 1

    for i in range(iterations):

        A = sigmoid(np.dot(W.T, X) + B)

        # cost function
        test = Y * np.log(A, where=A > 0)
        testt = (1 - Y) * np.log(1 - A, where=1 - A > 0)
        cost = -np.sum(test + testt)

        # Gradient Descent
        dW = (1 / n) * np.dot(X, (A - Y).T)
        dB = (1 / n) * np.sum(A - Y)

        W = W - learning_rate * dW
        B = B - learning_rate * dB
        print("cunt cost value:",cost)
        # Keeping track of our cost function value
        if cost <= 1e-2:
            break

    return W, B, cost


weight, bias, costList = logicstic_regisstion(x, y, learning_rate, iterations)

print('weigth:', weight)

result = sigmoid(np.dot(weight.T, testData) + bias)
print("done")
