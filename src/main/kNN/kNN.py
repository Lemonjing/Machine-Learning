#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib.pyplot as plt

"""
电影分类的例子
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
数据语预处理
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
数据语预处理
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
分类器错误率测试
"""
def datingClassTest():
    predictRatio = 0.10  # 预测样本所占比例
    datingDataMatrix, datingDataLabels = file2Matrix("../../resources/kNN/dating.txt")
    normMatrix, gap, minValue = autoNorm(datingDataMatrix)
    m = normMatrix.shape[0]
    numPredict = int(m * predictRatio)  # 用于训练的90%，用于预测的10%
    errorCount = 0.0
    for i in range(numPredict):
        classifierResult = classify0(normMatrix[i, :], normMatrix[numPredict:m, :]
                                     , datingDataLabels[numPredict:m], 3)
        print 'the classifier came back with: %d, the real answer is : %d' % (classifierResult, datingDataLabels[i])
        if (classifierResult != datingDataLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numPredict))


"""
分类器使用
约会网站预测
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

if __name__ == '__main__':
    test1()
    # datingClassTest()
    classiferPerson()  # 测试值10000, 10, 0.5
