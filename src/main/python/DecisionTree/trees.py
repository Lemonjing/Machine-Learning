#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import log
import operator

"""
计算香农熵
"""
def calcShannonEntropy(dataset):
    num = len(dataset)
    labelCounts = {}
    for line in dataset:
        currentLabel = line[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEntropy = 0.0
    for key in labelCounts:
        probability = float(labelCounts[key]) / num
        shannonEntropy -= probability * log(probability, 2)
    return shannonEntropy


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


"""
按照给定特征划分数据集
"""
def splitDataSet(dataset, axis, value):
    resultDataSet = []
    for line in dataset:
        if line[axis] == value:
            reducedLine = line[:axis]
            reducedLine.extend(line[axis+1:])
            resultDataSet.append(reducedLine)
    return resultDataSet

"""
选择最好的数据集划分方式
"""
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1  # 最后一列为label
    baseEntropy = calcShannonEntropy(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        features = [line[i] for line in dataset]
        uniqueVals = set(features)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            probability = len(subDataSet) / float(len(dataset))
            newEntropy += probability * calcShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
类标签不唯一时的多数表决
"""
def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
递归构造决策树-核心
"""
def createTree(dataSet, labels):
    classList = [line[-1] for line in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同时停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类别
        return majorityCount(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])
    featureVals = [line[bestFeature] for line in dataSet]
    uniqueVals = set(featureVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


"""
使用决策树进行分类
"""
def classify(inTree, labels, testVec):
    featureName = list(inTree.keys())[0]
    nextDict = inTree[featureName]
    featureIndex = labels.index(featureName)
    for key in nextDict:
        if testVec[featureIndex] == key:
            if type(nextDict[key]).__name__ == "dict":
                classLabel = classify(nextDict[key], labels, testVec)
            else:
                classLabel = nextDict[key]
    return classLabel


"""
持久化分类器
"""
def storeTree(inTree, fileName):
    import pickle
    fw = open(fileName, 'wb')
    pickle.dump(inTree, fw)
    fw.close()


def grabTree(fileName):
    import pickle
    fr = open(fileName, 'rb')
    return pickle.load(fr)


"""
准备测试数据
"""
def retrieveTree(i):
    myTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return myTrees[i]

def lensesTest():
    fr = open("../../resources/decisionTree/lenses.txt")
    lensesData = [line.strip().split("\t") for line in fr.readlines()]
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    lensesTree = createTree(lensesData, lensesLabels)
    print("lensesTree", lensesTree)
    import DecisionTree.treePlotter as tp
    tp.createPlot(lensesTree)

def test():
    dataSet, labels = createDataSet()
    print(calcShannonEntropy(dataSet))
    print(chooseBestFeatureToSplit(dataSet))
    # 决策树构造测试
    # print createTree(dataSet, labels)
    labels = ["no surfacing", "flippers"]
    # 分类测试
    print(classify(retrieveTree(0), labels, [1, 0]))
    # 持久化测试
    oldTree = createTree(dataSet, labels)
    print('oldTree', oldTree)
    storeTree(oldTree, 'tree.txt')
    newTree = grabTree('tree.txt')
    print('newTree', newTree)
    lensesTest()


if __name__ == '__main__':
    test()

