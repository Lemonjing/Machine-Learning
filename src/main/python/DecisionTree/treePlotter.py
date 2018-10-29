#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

"""
全局定义
"""
# 定义文本框和箭头格式[sawtooth 波浪方框,round4矩形方框,fc表示字体颜色的深浅0.1~0.9,依次变浅]
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrowArgs = dict(arrowstyle="<-")

# nodeTxt为要显示的文本，childPoint为指向的文本中心点，xy=parentPoint是箭头起始点坐标
# xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若未设置，默认为data
# va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')

# 取值
# ‘figure points’	points from the lower left of the figure
# ‘figure pixels’	pixels from the lower left of the figure
# ‘figure fraction’	fraction of figure from lower left
# ‘axes points’	points from lower left corner of axes
# ‘axes pixels’	pixels from lower left corner of axes
# ‘axes fraction’	fraction of axes from lower left
# ‘data’	use the coordinate system of the object being annotated (default)
# ‘polar’	(theta,r) if not native ‘data’ coordinates

"""
matplotlib注解
"""
def plotNode(nodeText, parentPoint, childPoint, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPoint, xycoords="axes fraction", xytext=childPoint
                            , textcoords="axes fraction", va="center", ha="center", bbox=nodeType, arrowprops=arrowArgs)

# """
# 测试画图
# """
# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=True)  # frameon表示是否绘制坐标轴矩形
#     plotNode(u"决策节点", (0.1, 0.5), (0.5, 0.1), decisionNode)
#     plotNode(u"叶子节点", (0.3, 0.8), (0.8, 0.1), leafNode)
#     plt.show()

"""
正式画图-主函数
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])  # 不显示坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # frameon表示是否绘制坐标轴矩形

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")

    plt.show()


"""
递归构造注解树
"""
def plotTree(myTree, parentPoint, nodeText):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    featureName = list(myTree.keys())[0]
    currentPoint = (plotTree.xOff + (1+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(parentPoint, currentPoint, nodeText)
    plotNode(featureName, parentPoint, currentPoint, decisionNode)
    nextDict = myTree[featureName]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in nextDict:
        # 递归画子树
        if type(nextDict[key]).__name__ == "dict":
            plotTree(nextDict[key], currentPoint, str(key))
        # 画叶结点
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(nextDict[key], currentPoint, (plotTree.xOff, plotTree.yOff), leafNode)
            plotMidText(currentPoint, (plotTree.xOff, plotTree.yOff), str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

"""
叶结点数目
"""
def getNumLeafs(myTree):
    numLeafs = 0
    rootFeature = list(myTree.keys())[0]
    rootFeatureDict = myTree[rootFeature]
    for key in rootFeatureDict.keys():
        if type(rootFeatureDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(rootFeatureDict[key])
        else:
            numLeafs += 1
    return numLeafs


"""
树的深度
"""
def getTreeDepth(myTree):
    maxDepth = 0
    rootFeature = list(myTree.keys())[0]
    rootFeatureDict = myTree[rootFeature]
    for key in rootFeatureDict:
        if type(rootFeatureDict[key]).__name__ == 'dict':
            tmpDepth = 1 + getTreeDepth(rootFeatureDict[key])
        else:
            tmpDepth = 1
        if tmpDepth > maxDepth:
            maxDepth = tmpDepth
    return maxDepth

"""
父子节点间添加文本
"""
def plotMidText(parentPoint, childPoint, midText):
    xMid = (parentPoint[0] - childPoint[0]) / 2.0 + childPoint[0]
    yMid = (parentPoint[1] - childPoint[1]) / 2.0 + childPoint[1]
    createPlot.ax1.text(xMid, yMid, midText)

"""
准备测试数据
"""
def retrieveTree(i):
    myTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return myTrees[i]

def test():
    createPlot(retrieveTree(1))


if __name__ == '__main__':
    test()
