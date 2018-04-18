#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet(fileName):
    dataList = []; labelList = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataList.append([float(lineArr[0]), float(lineArr[1])])
        labelList.append(float(lineArr[2]))
    dataArr = np.array(dataList)
    labelArr = np.array(labelList)
    print "dataArr", dataArr
    print "labelArr", labelArr
    return dataArr, labelArr

"""
i是第一个alpha的下标，m是alpha的数目
"""
def selectJrand(i, m):
    import random
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

"""
限制alphaJ在[H,L]之间
"""
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataArr, classLabels, C, toler, maxInter):
    dataMatrix = np.mat(dataArr) # m*n
    labelMatrix = np.mat(classLabels)  #m*1
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.zeros((m, 1))  # m*1
    iter = 0
    while (iter < maxInter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = np.multiply(alphas, labelMatrix) * (dataMatrix * dataMatrix[i,54:]) + b

def test():
    fileName = "../../resources/SVM/testSet.txt"
    loadDataSet(fileName)


if __name__ == '__main__':
    test()
