#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet():
    dataList = []; labelList = []
    fr = open("../../resources/Logistic/TestSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelList.append(int(lineArr[2]))
    dataArr = np.array(dataList)
    labelArr = np.array(labelList)
    return dataArr, labelArr

"""
激活函数，可替换为tanh
"""
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


"""
梯度上升方法
输出：参数的二维numpy数组
"""
def gradientAscent(inputArr, classArr):
    dataMatrix = np.mat(inputArr)  # 100*3
    labelMatrix = np.mat(classArr).transpose()  # 100*1
    m, n = np.shape(dataMatrix)  # 100, 3
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))  # 3*1 初始系数设为1
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 100*1
        error = (labelMatrix - h)  # 100*1
        # dataMatrix.transpose()*error为梯度
        weights = weights + alpha * dataMatrix.transpose()*error
    # 转化为numpy array以画图
    weights = weights.getA()
    return weights

"""
随机梯度上升
输出：参数的一维numpy数组
"""
def randomGradientAscent(inputArr, classArr):
    m, n = np.shape(inputArr)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(inputArr[i] * weights))
        error = classArr[i] - h
        weights = weights + alpha * error * inputArr[i]
    return weights


"""
改进的随机梯度上升
输出：参数的一维numpy数组
"""
def advancedRandomGradientAscent(inputArr, classArr, numIter=150):
    import random
    m, n = np.shape(inputArr)
    weights = np.ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4 / (1.0+i+j) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(inputArr[randIndex] * weights))
            error = classArr[randIndex] - h
            weights = weights + alpha * error * inputArr[randIndex]
            del(dataIndex[randIndex])
    return weights

"""
画出最佳拟合直线
输入weights为矩阵
"""
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataArr, labelArr = loadDataSet()
    n = np.shape(dataArr)[0]
    xcoord1 = []; ycoord1 = []
    xcoord2 = []; ycoord2 = []
    for i in range(n):
        if (labelArr[i]) == 1:
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

"""
用逻辑回归进行分类
"""
def classify(inX, weights):
    probability = sigmoid(sum(inX*weights))
    if probability > 0.5: return 1
    else: return 0

def colicTest():
    frTrain = open("../../resources/Logistic/HorseColicTraining.txt")
    frTest = open("../../resources/Logistic/HorseColicTest.txt")
    trainingSet= []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = advancedRandomGradientAscent(np.array(trainingSet),trainingLabels, 500)
    errorCount = 0; numTest = 0
    for line in frTest.readlines():
        numTest += 1
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if (classify(np.array(lineArr), trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTest
    print "the error rate of this test is: %f." % errorRate
    return errorRate

def multiTest():
    times = 10; errorSum = 0.0
    for i in range(times):
        errorSum += colicTest()
    print "after %d iterations the avg error rate is %f." % (times, errorSum/times)

def test():
    dataArr, labelArr = loadDataSet()
    # weights = gradientAscent(dataArr, labelArr)
    # weights = randomGradientAscent(dataArr, labelArr)
    # weights = advancedRandomGradientAscent(dataArr, labelArr)
    # plotBestFit(weights)
    multiTest()

if __name__ == '__main__':
    test()
