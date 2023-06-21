# x = (hours,sleeping,hours studying)


import numpy as np
from jsonpickle.ext import numpy
from numpy import genfromtxt
import csv


class neuralNetwork(object):
    def __init__(self, inputsize: int = 2, outputsize: int = 1, hiddensize: int = 3, epoch: int = 1000):
        # parameters
        self.inputSize: int = inputsize
        self.outputSize: int = outputsize
        self.hiddenSize: int = hiddensize
        self.epoch: int = epoch
        self.learningrate: np.float128 = 0.1

        self.weight_input_hidden = np.random.uniform(-0.5, 0.5, (self.hiddenSize, self.inputSize))
        self.weight_hidden_output = np.random.uniform(-0.5, 0.5, (self.outputSize, self.hiddenSize))

        self.bias_input_hidden = np.zeros((hiddensize, 1), dtype=np.float128)
        self.bias_output_hidden = np.zeros((outputsize, 1), dtype=np.float128)

        self.totalTestSample = None

    def sigmoid(self, s, deriv=False):
        return s * (1 - s) if deriv == True else 1 / (1 + np.exp(-s))

    def forward(self, x):

        # Forward propagation input -> hidden
        h_pre = self.bias_input_hidden + self.weight_input_hidden.dot(x)
        self.h = self.sigmoid(h_pre, deriv=False)
        # Forward propagation hidden -> output
        o_pre = self.bias_output_hidden + self.weight_hidden_output.dot(self.h)
        output = self.sigmoid(o_pre, deriv=False)
        return output

    def backWard(self, x, y, output: np):

        # Cost / Error calculation
        # err = 1 / len(output) * np.sum((output - y) ** 2, axis=0)
        current_nr_correct = int(np.argmax(output) == np.argmax(y))

        # Backpropagation output -> hidden (cost function derivative)
        outputDelta = output - y
        deltaweight = -self.learningrate * outputDelta.dot(self.h.T)
        self.weight_hidden_output += deltaweight
        biasUpdate = self.learningrate * outputDelta
        self.bias_output_hidden += -biasUpdate

        # Backpropagation hidden -> input (activation function derivative)
        delta_h = self.weight_hidden_output.T.dot(outputDelta) * self.sigmoid(self.h, deriv=True)
        self.weight_input_hidden += -self.learningrate * delta_h.dot(x.T)
        self.bias_input_hidden += -self.learningrate * delta_h
        return current_nr_correct

    def traing(self, x, y):
        labelList = set(y)
        self.totalTestSample = x.shape[0]
        self.encoding(labelList)
        for i in range(self.epoch):
            nr_correct = 0
            for index in range(x.shape[0]):
                sample = x[index]
                label = next(
                    np.copy(label['encoding']) for label in self.encodingSet if str(label['label']) == str(y[index]))
                sample.shape += (1,)
                label.shape += (1,)
                output = self.forward(sample)
                nr_correct += self.backWard(sample, label, output)
            print(f"Acc: {round((nr_correct / self.totalTestSample) * 100, 2)}%")

    def encoding(self, labelSet: set):
        self.encodingSet = []
        encodingList = np.eye(N=len(labelSet), M=self.outputSize)
        count: int = 0
        for label in labelSet:
            self.encodingSet.append({
                'label': label,
                'encoding': encodingList[count]
            })
            count += 1

    def acurrate(self):
        return round((self.nr_correct / self.totalTestSample) * 100, 2)

    def checking(self, testData):
        resultList = []
        for sample in testData:
            sample.shape += (1,)
            o = self.forward(sample)
            resultList.append(self.encodingSet[o.argmax()]['label'])
        return resultList


trainData = genfromtxt('lab3_data/lab3_train.csv', delimiter=',', skip_header=1, dtype=np.float128)
outputData = trainData[:, 0]  # for last column
inputData = np.delete(trainData, 0, 1)  # delete second column of C # for all but last column

NN = neuralNetwork(inputsize=784,
                   hiddensize=342,
                   outputsize=4,
                   epoch=1000)
NN.traing(inputData, outputData)
testData = genfromtxt('lab3_data/lab3_test.csv', delimiter=',', skip_header=1, dtype=np.float128)
result = NN.checking(testData)

with open("test_ans.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(result)
# np.savetxt("test_ans.csv", np.asarray(result))

# Show results
