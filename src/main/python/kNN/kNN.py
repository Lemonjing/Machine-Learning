#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir


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
        print('the classifier came back with: %d, the real answer is : %d' % (classifierResult, realDigit))
        if (classifierResult != realDigit):
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / mTest))


if __name__ == '__main__':
    # datingClassTest()
    # classiferPerson()  # 测试值10000, 10, 0.5
    handWritingClassTest()
