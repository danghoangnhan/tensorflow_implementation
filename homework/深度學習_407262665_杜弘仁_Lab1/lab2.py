# II.實驗二
# 1.程式輸入需使用讀檔方式，輸出需至少print 出w1, w2, b 和測試準確率的數值
# 2.共有一組訓練資料檔名為Iris_training 及測試資料檔名為Iris_test，格式皆為.txt。
# i.訓練資料中，三個欄位分別代表x1, x2, y(label)，三個數值以逗號相隔。e.g., 5.7,3,-1
# ii.測試資料中，三個欄位分別代表x1, x2, y(label)，三個數值以逗號相隔。e.g., 7.7,3.8,-1
# III.程式畫圖請依訓練及測試資料標上座標點，以及最後結果的線條
import numpy as np

b = np.random.randint(0, 1)
traningData = np.loadtxt('Iris_training.txt', usecols=range(3), delimiter=",")
testData = np.loadtxt('Iris_test.txt', usecols=range(3), delimiter=",")
testSample = testData[:, [0, 1]]
maxEpoch = 1000
w = np.array([1 for _ in range(len(traningData[0])-1)])


def sign(current_w, current_x, current_b):
    return np.sign(current_x[0] * current_w[0] + current_x[1] * current_w[1] + current_b)



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


(w, b) = perceptron(traningData[:, [0, 1]], traningData[:, 2], w, b)
print("weight vector:", w)
print("bias:", b)
print("classifier of data is:")
for i in range(len(testSample)):
    print(sign(w, testSample[i], b))



