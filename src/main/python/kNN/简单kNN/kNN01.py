#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator

"""
电影分类的例子（收集数据）
四组特征及其类别
"""
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
