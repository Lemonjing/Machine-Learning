#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

"""
电影分类的例子（收集数据）
"""
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
电影分类的例子测试
"""
def test1():
    group, labels = createDataSet()
    print "group=", str(group)
    print "labels=", str(labels)
    print classify0([0, 0], group, labels, 3)


"""
改进约会网站配对的matplotlib画图测试
"""
def datingTest():
    datingMatrix, datingLabels = file2Matrix("../../resources/kNN/dating.txt")
    print "datingMatrix=", datingMatrix
    print "datingLabels=", datingLabels

    def matplotGraph():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_title(u"子图")
        ax.scatter(datingMatrix[:, 1], datingMatrix[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
        plt.title(u"约会网站配对效果")
        plt.xlabel(u"玩视频游戏所耗时间百分比")
        plt.ylabel(u"每周消费的冰淇淋公升数")
        plt.show()

    matplotGraph()


"""
kNN分类核心算法
"""
def classify0(inX, dataset, lables, k):
    """
    距离计算
    """
    datasetSize = dataset.shape[0]  # 数据集行数
    diffMatrix = np.tile(inX, (datasetSize, 1)) - dataset
    sqDiffMatrix = diffMatrix ** 2
    sqDistance = sqDiffMatrix.sum(axis=1)  # axis=1表示按行相加,axis=0表示按列相加
    distances = sqDistance ** 0.5
    sortedDistanceIndex = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = lables[sortedDistanceIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 排序规则,reverse=True 降序,reverse=False升序（默认)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


"""
数据预处理（准备数据）
1.文件转矩阵
"""
def file2Matrix(fileName):
    fr = open(fileName)
    lines = fr.readlines()
    fileLen = len(lines)
    matrix = np.zeros((fileLen, 3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        lineArr = line.split("\t")
        matrix[index, :] = lineArr[0:3]
        labels.append(int(lineArr[-1]))
        index += 1
    return matrix, labels


"""
数据语预处理（准备数据）
2.数据集归一化特征值
"""
def autoNorm(dataset):
    minValue = dataset.min(0)  # axis=0; 每列的最小值
    maxValue = dataset.max(0)
    gap = maxValue - minValue
    normDataSet = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - np.tile(minValue, (m, 1))
    normDataSet = normDataSet / np.tile(gap, (m, 1))

    return normDataSet, gap, minValue


"""
改进约会网站配对分类器错误率测试（测试）
"""
def datingClassTest():
    testRatio = 0.10  # 预测样本所占比例
    datingDataMatrix, datingDataLabels = file2Matrix("../../resources/kNN/dating.txt")
    normMatrix, gap, minValue = autoNorm(datingDataMatrix)
    m = normMatrix.shape[0] # 样本总大小
    mTest = int(m * testRatio)  # 用于预测的占10%
    errorCount = 0.0
    for i in range(mTest):
        classifierResult = classify0(normMatrix[i, :], normMatrix[mTest:m, :]
                                     , datingDataLabels[mTest:m], 3)
        print 'the classifier came back with: %d, the real answer is : %d' % (classifierResult, datingDataLabels[i])
        if (classifierResult != datingDataLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(mTest))


"""
改进约会网站配对分类器使用（预测）
"""
def classiferPerson():
    resultList = [u"一点也没", u"有些魅力", u"极具魅力"]
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    percentTime = float(raw_input("percentage of time spent playing games?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMatrix, datingDataLabels = file2Matrix("../../resources/kNN/dating.txt")
    normMatrix, gap, minValue = autoNorm(datingDataMatrix)
    inArr = np.array([ffMiles, percentTime, iceCream])
    # 注意这里的预测输入inArr也要归一化
    classifierResult = classify0((inArr - minValue) / gap, normMatrix, datingDataLabels, 3)
    print "you probably like this person:", resultList[classifierResult - 1]


"""
手写数字识别文本图像转矩阵（准备数据）
"""
def img2Vector(fileName):
    vector = np.zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0, 32*i + j] = lineStr[j]
    return vector


"""
手写数字识别错误率测试（测试）
"""
def handWritingClassTest():
    labels = []
    trainingFiles = listdir("../../resources/kNN/trainingDigits")
    m = len(trainingFiles)
    trainingMatrix = np.zeros((m, 1024))
    for i in range(m):
        fileName = trainingFiles[i]
        targetDigit = int(fileName.split("_")[0])
        labels.append(targetDigit)
        trainingMatrix[i, :] = img2Vector("../../resources/kNN/trainingDigits/%s" % fileName)

    testFiles = listdir("../../resources/kNN/testDigits")
    mTest = len(testFiles)
    errorCount = 0.0
    for i in range(mTest):
        fileName = testFiles[i]
        realDigit = int(fileName.split("_")[0])
        testVector = img2Vector("../../resources/kNN/testDigits/%s" % fileName)
        classifierResult = classify0(testVector, trainingMatrix, labels, 3)
        print 'the classifier came back with: %d, the real answer is : %d' % (classifierResult, realDigit)
        if (classifierResult != realDigit):
            errorCount += 1.0
    print "the total number of errors is: %d" % errorCount
    print "the total error rate is: %f" % (errorCount / mTest)



if __name__ == '__main__':
    test1()
    # datingClassTest()
    # classiferPerson()  # 测试值10000, 10, 0.5
    handWritingClassTest()
