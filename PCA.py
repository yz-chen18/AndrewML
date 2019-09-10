import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
    dataArr = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        line = line.strip().split()
        dataArr.append([float(data) for data in line])
    return np.array(dataArr)

def draw(datamat, reconMat):
    x1 = [data[0] for data in datamat]
    y1 = [data[1] for data in datamat]
    x2 = [data[0] for data in reconMat]
    y2 = [data[1] for data in reconMat]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x1, y1, color = 'b')
    ax.scatter(x2, y2, color = 'k')

    plt.show()

def pca(datamat, N):
    meanVal = np.mean(datamat, axis = 0)
    data = datamat - meanVal
    covMat = np.cov(data, rowvar = 0)

    eigVals, eigVect = np.linalg.eig(covMat)
    eigInd = np.argsort(eigVals)[:-(N + 1):-1]

    projMat = eigVect[:, eigInd]
    lowDDataMat = data.dot(projMat)
    reconMat = lowDDataMat.dot(projMat.T) + meanVal

    return reconMat, projMat

datamat = loadData("C:\\Users\\91969\\Desktop\\ML\\machinelearninginaction\\Ch13\\testSet.txt")
reconMat, projMat = pca(datamat, 1)
draw(datamat, reconMat)
