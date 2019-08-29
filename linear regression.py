import numpy as np
import matplotlib.pyplot as plt

def regression(inputMatrix):
    shape = inputMatrix.shape
    m = shape[0]     #实例个数
    n = shape[1] - 1 #特征数量，需要减去输入矩阵中的实际值
    xMat = []
    yMat = []
    J = []
    for i in range(m):
        data = [1]
        yMat.append([inputMatrix[i][-1]]) #将每一个data中的实际值录入
        data.extend(list(inputMatrix[i][:-1]))

        xMat.append(data)

    alpha = 0.001
    xMat = np.array(xMat)
    yMat = np.array(yMat)
    theta = (np.random.normal(size = (n + 1, 1)))

    turn = 20000 #循环次数

    #linear regression算法关键，实现向量化
    for i in range(turn):\
        #可进行cost计算
        """
        J_theta = (np.dot(xMat, theta) - yMat).T.dot(np.dot(xMat, theta) - yMat) / (2*m)
        J.append(J_theta)
        """
        delta = xMat.T.dot(np.dot(xMat, theta) - yMat)
        theta = theta - alpha / m * delta


    return theta
