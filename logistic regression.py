import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(x))

def regression(inputMatrix):
    m = len(inputMatrix)        #实例个数
    n = len(inputMatrix[0]) - 2 #特征个数，减去标签(1或0)
    xMat = []
    labels = []
    for i in range(m):
        xMat.append(inputMatrix[i][:-1])
        labels.append([inputMatrix[i][-1]])

    xMat = np.array(xMat)
    labels = np.array(labels)
    theta = np.ones((n + 1, 1))

    turn = 1000 #循环次数
    alpha = 0.01

    #logistic regression算法实现关键
    for n in range(turn):
        delta = xMat.T.dot(sigmoid(np.dot(xMat, theta)) - labels)
        theta = theta - alpha / m * delta

    return theta


#下为利用matplotlib进行决策边界的绘制与描点
"""
def draw():
    x = list(np.random.randint(-5, 5, 10))
    y = list(np.random.randint(-5, 5, 10))
    matrix = []

    for n in range(len(x)):
        label = random.randint(0, 1)
        matrix.append([x[n], y[n], label])
        if (label == 0):
            plt.scatter(x[n], y[n], c = "r")
        else:
            plt.scatter(x[n], y[n], c = "b")

    print(matrix)
    theta = regression(matrix)

    x = [n for n in range(-5, 6)]
    y = [theta[0] + theta[1]*n for n in x]
    plt.plot(x, y)

    plt.show()
"""