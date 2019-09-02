import numpy as np

def sigmoid(inX):
    return 1/(1 + np.exp(-inX))

def gradCheck():
    return

def NNlearning(matrix, neuNum, depth, turn):
    inX = []
    Y = []
    layerList = []
    for arr in matrix:
        inX.append(matrix[:-1])
        Y.append([matrix[-1]])
    inX = np.array(inX)
    Y = np.array(Y)
    m = len(inX)
    n = len(inX[0])
    for i in range(depth + 1):
        if (n == 0):
            layer = np.random.normal((neuNum, m + 1))
            layerList.append(layer)
        elif (n == depth):
           layer = np.random.normal((1, neuNum))
           layerList.append(layer)
        else:
            layer = np.random.normal((neuNum,neuNum))
            layerList.append(layer)

    for i in range(turn):
        backProp(inX, Y, layerList)
    return

def forProp(inX, layerList):
    mat = []
    for layer in layerList:
        mat = inX.dot(layer.T)
        inX = np.exp(mat)
        mat.append(inX)
    return mat

def backProp(inX, Y, layerList):
    delta = []
    Delta = []

    for i in range(1, len(layerList)):
        Delta.append(np.zeros(layerList[i].shape))

    for m in inX:
        aMat = forProp(m, layerList)
        y = aMat[-1]

        for i in range(len(layerList)):
            if (i == 0):
                d = y - Y
                delta.insert(0, d)

            else:
                d = layerList[-i].T.dot(delta[0]).dot(aMat[-(i + 1)].dot(np.ones(aMat.T.shape) - aMat.T))
                delta.insert(0, d)


        for l in range(len(delta)):
            Delta[l] = Delta[l] + aMat[l].dot(delta[l]) #i dont know what to do next
    return