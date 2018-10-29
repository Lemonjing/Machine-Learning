#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib.pyplot as plt

RESOURCES_ROOT = "/Users/chucheng/GitHub/MachineLearning/src/main/resources"

"""
改进约会网站配对的matplotlib画图测试
"""
def showData():
    datingMatrix, datingLabels = file2Matrix(RESOURCES_ROOT + "/kNN/dating.txt")

    def matplotGraph():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(datingMatrix[:, 1], datingMatrix[:, 2], s=15.0 * np.array(datingLabels), c=15.0 * np.array(datingLabels))
        plt.title(u"约会网站配对效果")
        plt.xlabel(u"玩视频游戏所耗时间百分比")
        plt.ylabel(u"每周消费的冰淇淋公升数")
        plt.show()

    matplotGraph()


"""
kNN分类核心算法
  Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
  Returns:
sortedClassCount[0][0] - 分类结果
"""
def classify0(inX, dataset, lables, k):
    # 距离计算
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
    # key=operator.itemgetter(1)等于lambda x: x[1]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
数据预处理（准备数据）
1.文件转矩阵
"""
def file2Matrix(fileName):
    fr = open(fileName)
    lines = fr.readlines()
    numberOfLines = len(lines)
    matrix = np.zeros((numberOfLines, 3))
    labels = []
    index = 0
    for line in lines:
        lineArr = line.strip().split("\t")
        matrix[index, :] = lineArr[0:3]
        labels.append(int(lineArr[-1]))
        index += 1
    return matrix, labels


"""
数据语预处理（准备数据）
2.数据集归一化特征值
"""
def autoNorm(dataset):
    minValue = dataset.min(axis=0)  # axis=0; 每列
    maxValue = dataset.max(axis=0)  # axis=0; 每列
    gap = maxValue - minValue
    normDataSet = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - np.tile(minValue, (m, 1))
    normDataSet = normDataSet / np.tile(gap, (m, 1))
    return normDataSet, gap, minValue


"""
改进约会网站配对分类器错误率测试
"""
def datingClassTest():
    testRatio = 0.10  # 预测样本所占比例
    datingDataMatrix, datingDataLabels = file2Matrix(RESOURCES_ROOT + "/kNN/dating.txt")
    normMatrix, gap, minValue = autoNorm(datingDataMatrix)
    m = normMatrix.shape[0]  # 样本总大小
    numOfTest = int(m * testRatio)  # 用于预测的占10%
    errorCount = 0.0
    for i in range(numOfTest):
        classifierResult = classify0(normMatrix[i, :], normMatrix[numOfTest:m, :]
                                     , datingDataLabels[numOfTest:m], 3)
        print ("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingDataLabels[i]))
        if (classifierResult != datingDataLabels[i]):
            errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount / float(numOfTest)))


"""
改进约会网站配对分类器使用
"""
def classiferPerson():
    resultList = [u"一点也没", u"有些魅力", u"极具魅力"]
    ffMiles = float(input("Frequent flier miles earned per year?"))
    percentTime = float(input("Percentage of time spent playing games?"))
    iceCream = float(input("Liters of ice cream consumed per year?"))
    datingDataMatrix, datingDataLabels = file2Matrix(RESOURCES_ROOT + "/kNN/dating.txt")
    normMatrix, gap, minValue = autoNorm(datingDataMatrix)
    inArr = np.array([ffMiles, percentTime, iceCream])
    # 注意这里的预测输入inArr也要归一化
    classifierResult = classify0((inArr - minValue) / gap, normMatrix, datingDataLabels, 3)
    print ("You probably like this person:", resultList[classifierResult - 1])


if __name__ == '__main__':
    showData()
    # datingClassTest()
    # classiferPerson()
