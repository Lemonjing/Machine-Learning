#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet():
    dataMatrix = []; labelMatrix = []
    fr = open("../../resources/Logistic/TestSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMatrix.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMatrix.append(int(lineArr[2]))
    return dataMatrix, labelMatrix

"""
激活函数，可替换为tanh
"""
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


"""
梯度上升方法
"""
def gradientAscent(inputMatrix, classLabels):
    dataMatrix = np.mat(inputMatrix)  # 100*3
    labelMatrix = np.mat(classLabels).transpose()  # 100*1
    m, n = np.shape(dataMatrix)  # 100, 3
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))  # 3*1
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 100*1
        error = (labelMatrix - h)  # 100*1
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

"""
画出最佳拟合直线
输入weights为矩阵
"""
def plotBestFit(weights):
    # 转化为numpy array
    weights = weights.getA()
    import matplotlib.pyplot as plt
    dataMatrix, labelMatrix = loadDataSet()
    dataArr = np.array(dataMatrix)
    n = np.shape(dataArr)[0]
    xcoord1 = []; ycoord1 = []
    xcoord2 = []; ycoord2 = []
    for i in range(n):
        if (labelMatrix[i]) == 1:
            xcoord1.append(dataArr[i, 1]); ycoord1.append(dataArr[i, 2])
        else:
            xcoord2.append(dataArr[i, 1]); ycoord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord1, ycoord1, s=30, c='red', marker='s')
    ax.scatter(xcoord2, ycoord2, s=30, c='green', marker='o')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    ax.set_title(u"梯度上升逻辑回归最佳拟合直线")
    plt.xlabel("X1"); plt.ylabel("X2")
    plt.show()

def test():
    dataMatrix, labelMatrix = loadDataSet()
    weights = gradientAscent(dataMatrix, labelMatrix)
    print "weights", weights
    plotBestFit(weights)

if __name__ == '__main__':
    test()
