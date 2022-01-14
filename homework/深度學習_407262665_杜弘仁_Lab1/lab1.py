import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

trainningData = np.array([[1, 0],
                          [1, 3],
                          [2, -6],
                          [-1, -3],
                          [-5, 5],
                          [5, 2],
                          [-2, 2],
                          [-7, 2],
                          [4, -4],
                          [5, -1]])
y = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, -1])
w = np.array([1, 1])
b_init = 0
maxEpoch = 1000
testTingData = np.array([[2, -4],
                         [-5, 1],
                         [-2, -2]])


def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    x10 = -w0 / w1
    return plt.plot([x10, x10], [-100, 100], 'k')


def sign(current_w, current_x, current_b):
    return np.sign(current_x[0] * current_w[0] + current_x[1] * current_w[1] + current_b)


def devideGroup(current_w, currentX):
    groupOne = []
    groupTwo = []
    for i in range(len(currentX)):
        if sign(current_w, currentX[i], b) == 1:
            groupOne.append(currentX)
        else:
            groupTwo.append(currentX)
    return groupOne, groupTwo


def perceptron(current_x, current_y, current_w, current_b):
    n_iteration = 0
    while True and n_iteration < maxEpoch:
        isFinal = True
        n_iteration += 1
        for i in range(len(current_x)):
            if sign(current_w, current_x[i], current_b) != current_y[i]:
                isFinal = False
                current_b = current_b * current_y[i]
                w[0] = w[0] + current_y[i] * current_x[i][0]
                w[1] = w[1] + current_y[i] * current_x[i][1]
        if isFinal:
            break
    return current_w, current_b


(w, b) = perceptron(trainningData, y, w, b_init)
trainingGroupOne, trainingGroupTwo = devideGroup(w, trainningData)
testingGroupOne, testTingGroupTwo = devideGroup(w, testTingData)

print("weight vector:", w)
print("bias:", b)

print("classifier of data is:")
for i in range(len(testTingData)):
    print(sign(w, testTingData[i], b))
