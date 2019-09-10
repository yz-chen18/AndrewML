import numpy as np
import matplotlib.pyplot as plt

def loadData(fpath):
    mat = []
    labels = []
    with open(fpath, 'r') as f:
        for line in f.readlines():
            label = line.split()[-1]
            line = line.split()[:-1]
            line = [float(num) for num in line]
            mat.append(line)
            labels.append(int(label))

    mat = np.array(mat)
    return mat, labels

def choose2rand(m):
    p1 = m
    p2 = m
    while (p1 == p2):
        p1 = np.random.randint(0, m)
        p2 = np.random.randint(0, m)

    return p1, p2

def k_means(inX, maxIter, K = 2, looptimes = 1):
    m = len(inX)
    times = 0
    labelList = []
    costList  = []
    labelMat  = [1 for i in range(m)]
    p0, p1    = choose2rand(m)
    p0        = inX[p0]
    p1        = inX[p1]


    for l in range(looptimes):
        p0, p1 = choose2rand(m)
        p0 = inX[p0]
        p1 = inX[p1]
        templabel = [0 for i in range(m)]
        for i in range(maxIter):
            for j in range(m):
                d0 = np.multiply(inX[j] - p0, inX[j] - p0).sum()
                d1 = np.multiply(inX[j] - p1, inX[j] - p1).sum()

                if (d0 > d1):
                    templabel[j] = 1

            p1_sum = np.zeros(inX[0].shape)
            p1_num = 0
            p2_sum = np.zeros(inX[0].shape)
            p2_num = 0
            for p in range(m):
                if (templabel[p] == 0):
                    p1_sum += inX[p]
                    p1_num += 1

                else:
                    p2_sum += inX[p]
                    p2_num += 1

            p0 = p1_sum / p1_num
            p1 = p2_sum / p2_num

        cost = 0
        for i in range(m):
            if (templabel[i] == 0):
                d = np.multiply(inX[i] - p0, inX[i] - p0).sum()
            else:
                d = np.multiply(inX[i] - p1, inX[i] - p1).sum()

            cost += d

        labelList.append(templabel)
        costList.append(cost)

    mincost = costList[0]
    minIndex = 0
    for i in range(len(costList)):
        if (costList[i] < mincost):
            minIndex = i

    labelMat = labelList[minIndex]

    return labelMat, p0, p1

def draw(fpath):
    inX, labels = loadData(fpath)
    labelMat, p1, p2 = k_means(inX, 100, 2, 4)
    fig = plt.figure()
    m = len(labelMat)
    error = 0
    rate = 0

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(p1[0], p1[1], color = 'k', s = 50)
    ax.scatter(p2[0], p2[1], color = 'k', s = 50)

    for i in range(m):
        if (labelMat[i] != labels[i]):
            error += 1
        if (labelMat[i] == 1):
            ax.scatter(inX[i][0], inX[i][1], color='r', s = 20)

        else:
            ax.scatter(inX[i][0], inX[i][1], color='b', s = 20)

    #print(labelMat)
    #print(labels)
    plt.show()
    rate = error / m
    if (rate > 0.5):
        rate = 1 - rate
    print(rate)
    '''
    for j in range(4):
        labelMat = labelList[j]
        axj = fig.add_subplot(2, 2, j + 1)
        for i in range(m):
            if (labelMat[i] == 1):
                axj.scatter(inX[i][0], inX[i][1], color = 'r')

            else:
                axj.scatter(inX[i][0], inX[i][1], color = 'b')

    plt.show()
    '''

draw("C:\\Users\\91969\\Desktop\\testSet.txt")