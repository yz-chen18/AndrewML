import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def regression(inputMatrix):
    m = len(inputMatrix)        #实例个数
    n = len(inputMatrix[0]) - 1 #特征个数，减去标签(1或0)
    xMat = []
    labels = []
    for i in range(m):
        data = [1]
        data.extend(inputMatrix[i][:-1])
        labels.append([inputMatrix[i][-1]])
        xMat.append(data)

    xMat = np.array(xMat)
    labels = np.array(labels)
    theta = np.random.uniform(0, 1, (n + 1, 1))

    turn = 5000 #循环次数
    alpha = 0.01

    #logistic regression算法实现关键
    for n in range(turn):
        delta = xMat.T.dot(sigmoid(np.dot(xMat, theta)) - labels)
        #print(labels)
        theta = theta - alpha / m * delta

    return theta


#下为利用matplotlib进行决策边界的绘制与描点
def draw():
    fpath = "C:\\Users\\91969\\Desktop\\testSet.txt"
    f = open(fpath, "r")
    x = []
    y = []
    for n in f.readlines():
        xarr = [float(i) for i in n.split()[:-1]]
        yarr = [int(i) for i in n.split()[-1]]
        x.append(xarr)
        y.append(yarr)

    matrix = []

    for n in range(len(x)):
        if (y[n][0] == 0):
            matrix.append([x[n][0], x[n][1], 0])
            plt.scatter(x[n][0], x[n][1], c = "r")
        else:
            matrix.append([x[n][0], x[n][1], 1])
            plt.scatter(x[n][0], x[n][1], c = "b")

    print(matrix)
    theta = regression(matrix)

    x = [n for n in range(-3, 4)]
    y = [(-theta[0] - theta[1]*n)/theta[2] for n in x]
    plt.plot(x, y)
    print(theta)
    plt.show()

draw()