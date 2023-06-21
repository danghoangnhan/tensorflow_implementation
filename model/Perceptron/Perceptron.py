# import packages
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Perceptron:
    # defines the constructor to our Perceptron class
    def __init__(self, N, eta=0.0001,bias=1):
        # N: The number of columns in our input feature vectors
        # eta: learning rate for the Perceptron algorithm. We’ll set this value to 0.01 by default.
        # Common choices of learning rates are normally in the range a = 0:1;0:01;0:001.
        # initialize weight matrix and store the learning rate
        # Note: - using a "normal distribution"
        #      - N+1 (+1 for the bias param)
        #      - Divide W by the square-root of the number of inputs. Common
        #           technique for scaling our weight matrix (faster convergence)
        self.W = np.random.randn(N) / np.sqrt(N)
        self.eta = eta
        self.bias = bias
    def step(self, x):
        # apply the step function
        return np.where(x > 0, 1, 0)


    def fit(self, X_train, y_train,X_val,y_val, epochs):
        # loop over the desired number of epochs
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_val = labelencoder.fit_transform(y_val)

        for epoch in np.arange(0, epochs):
            # loop over each individual data point:
            for (x, target) in zip(X_train, y_train):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction 
                prediction = self.step(np.dot(x, self.W)+ self.bias)                 
                self.W +=  + self.eta * (target - prediction) * x
                self.bias +=  self.eta * (target - prediction) * 1

            predictions = self.predict(X_val)
            correct_predictions = np.sum(predictions == y_val)
            accuracy = correct_predictions / len(y_val)
            print("accuracy:",accuracy)  
    
    def predict(self, X):
        return self.step(np.dot(X, self.W)+ self.bias)
