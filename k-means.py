import numpy as np
import matplotlib.pyplot as plt

def loadData(fpath):
    mat = []
    with open(fpath, 'r') as f:
        for line in f.readlines():
            line = line.split()[:-1]
            line = [float(num) for num in line]
            mat.append(line)

    mat = np.array(mat)
    return mat

def choose2rand(m):
    p1 = m
    p2 = m
    while (p1 == p2):
        p1 = np.random.randint(0, m)
        p2 = np.random.randint(0, m)

    return p1, p2

def k_means(inX, maxIter, K = 2):
    m = len(inX)
    labelMat = [1 for i in range(m)]
    p1, p2 = choose2rand(m)
    p1 = inX[p1]
    p2 = inX[p2]


    for i in range(maxIter):
        for j in range(m):
            d1 = np.multiply(inX[j] - p1, inX[j] - p1).sum()
            d2 = np.multiply(inX[j] - p2, inX[j] - p2).sum()

            if (d1 > d2):
                labelMat[j] = 2

        p1_sum = np.zeros(inX[0].shape)
        p1_num = 0
        p2_sum = np.zeros(inX[0].shape)
        p2_num = 0
        for p in range(m):
            if (labelMat[p] == 1):
                p1_sum += inX[p]
                p1_num += 1

            else:
                p2_sum += inX[p]
                p2_num += 1

        p1 = p1_sum / p1_num
        p2 = p2_sum / p2_num

    return labelMat

def draw(fpath):
    inX = loadData(fpath)
    labelMat = k_means(inX, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    m = len(labelMat)

    for i in range(m):
        if (labelMat[i] == 1):
            ax1.scatter(inX[i][0], inX[i][1], color = 'r')

        else:
            ax1.scatter(inX[i][0], inX[i][1], color = 'b')

    plt.show()

draw("C:\\Users\\91969\\Desktop\\testSet.txt")