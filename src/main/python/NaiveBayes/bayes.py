#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

# 项目案例1：屏蔽社区留言板的侮辱性言论
# 项目案例2：垃圾邮件过滤
"""
创建数据集
return 单词列表postingList, 所属类别classVec
"""
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

"""
所有文档出现的不重复词列表
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

"""
词集模型
输入文档的单词在词汇表是否出现, 输入转化为词向量
"""
def setOfWords2Vec(vocabList, inputSet):
    res = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            res[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return res

"""
词袋模型
输入文档的单词在词汇表是否出现及次数, 输入转化为词向量
"""
def bagOfWords2Vec(vocabList, inputSet):
    res = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            res[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return res

"""
朴素贝叶斯核心函数
原始版本
"""
def _trainNB0(trainMatrix, trainClass):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    print("numWords", numWords)
    pAbusive = sum(trainClass) / float(numTrainDocs)  # P(class=1)侮辱性文档概率
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainClass[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    print("p0Denom", p0Denom)
    print("p1Denom", p1Denom)
    p0Vec = p0Num / p0Denom
    p1Vec = p1Num / p1Denom
    return p0Vec, p1Vec, pAbusive

"""
朴素贝叶斯核心函数
解决概率为0的乘积问题和下溢问题的改进版本
"""
def trainNB0(trainMatrix, trainClass):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainClass) / float(numTrainDocs)  # P(class=1)侮辱性文档概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainClass[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        else:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)
    return p0Vec, p1Vec, pAbusive

"""
朴素贝叶斯分类函数
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)  # 实际是求了贝叶斯的分子，因为分母是一致的
    p0 = sum(vec2Classify * p0Vec) + math.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
测试朴素贝叶斯分类
"""
def testNB():
    docs, classes = loadDataSet()
    vocabulary = createVocabList(docs)
    trainMatrix = []
    for doc in docs:
        trainMatrix.append(setOfWords2Vec(vocabulary, doc))
    p0V, p1V, pAb = trainNB0(trainMatrix, classes)
    testDoc = ['love', 'my', 'dalmation']
    testDocVec = setOfWords2Vec(vocabulary, testDoc)
    print(testDoc, "classified as: ", classifyNB(testDocVec, p0V, p1V, pAb))
    testDoc2 = ['stupid', 'garbage']
    testDoc2Vec = setOfWords2Vec(vocabulary, testDoc2)
    print(testDoc2, "classified as: ", classifyNB(testDoc2Vec, p0V, p1V, pAb))

"""
使用朴素贝叶斯进行交叉验证
"""
def textParse(longText):
    import re
    words = re.split(r"\W*", longText)
    return [word.lower() for word in words if len(word) > 2]

"""
垃圾邮件过滤错误率测试
"""
def spamTest():
    import random
    docs = []; classes = []; fullText = []
    for i in range(1, 26):
        words = textParse(open("../../resources/NaiveBayes/email/spam/%d.txt" % i).read())
        docs.append(words)
        fullText.extend(words)
        classes.append(1)
        words = textParse(open("../../resources/NaiveBayes/email/ham/%d.txt" % i).read())
        docs.append(words)
        fullText.extend(words)
        classes.append(0)
    vocabulary = createVocabList(docs)
    trainSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMatrix = []; trainClasses = []
    for docIndex in trainSet:
        trainMatrix.append(setOfWords2Vec(vocabulary, docs[docIndex]))
        trainClasses.append(classes[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMatrix, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabulary, docs[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classes[docIndex]:
            errorCount += 1
            print("classification error", docs[docIndex])
    print("the error rate is ", float(errorCount) / len(testSet))

"""
朴素贝叶斯从个人广告中获取区域倾向
移除高频词
"""
def calcMostFreq(vocabulary, fullText):
    import operator
    freqDict = {}
    for word in vocabulary:
        freqDict[word] = fullText.count(word)
    # 排序规则,reverse=True 降序,reverse=False升序（默认)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # 5个高频词
    return sortedFreq[:5]

"""
移除停用词
"""
def stopWords():
    import re
    inputWords = open("../../resources/NaiveBayes/stopword.txt").read()
    words = re.split(r"\W*", inputWords)
    stopWords = [word.lower() for word in words if len(word) > 0]
    return stopWords

"""
从rss源获取广告倾向feed[i]属于第i类的概率,feed[i]可理解为某一地域
"""
def localWords(feed0, feed1):
    import random
    docs = []; classes = []; fullText = []
    minLen = min(len(feed0["entries"]), len(feed1["entries"]))

    # feed0是第0类 feed1是第1类
    for i in range(minLen):
        words = textParse(feed0["entries"][i]["summary"])
        docs.append(words)
        fullText.extend(words)
        classes.append(0)
        words = textParse(feed1["entries"][i]["summary"])
        docs.append(words)
        fullText.extend(words)
        classes.append(1)
    vocabulary = createVocabList(docs)
    # 停用词移除
    for word in stopWords():
        if word in vocabulary:
            vocabulary.remove(word)
    # 高频词移除
    top30Words = calcMostFreq(vocabulary, fullText)
    for tuple in top30Words:
        if tuple[0] in vocabulary: vocabulary.remove(tuple[0])

    trainSet = list(range(2 * minLen)); testSet = []

    # 选2篇为测试集合
    for i in range(2):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMatrix = []; trainClasses = []
    for docIndex in trainSet:
        trainMatrix.append(bagOfWords2Vec(vocabulary, docs[docIndex]))
        trainClasses.append(classes[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMatrix, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2Vec(vocabulary, docs[docIndex])
        if classifyNB(wordVec, p0V, p1V, pSpam) != classes[docIndex]:
            errorCount += 1
    print("the error rate is ", float(errorCount) / len(testSet))
    return vocabulary, p0V, p1V

"""
显示地域相关用词，nasa第一类，yahoo第二类
"""
def getTopWords(nasa, yahoo):
    vocabulary, p0V, p1V = localWords(nasa, yahoo)
    topNasa = []; topYahoo = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topNasa.append((vocabulary[i], p0V[i]))
        if p1V[i] > -6.0: topYahoo.append((vocabulary[i], p1V[i]))
    sortedNasa = sorted(topNasa, key=lambda pair: pair[1], reverse=True)
    sortedYahoo = sorted(topYahoo, key=lambda pair: pair[1], reverse=True)
    print("======sortedNasa======")
    for item in sortedNasa:
        print(item[0])
    print("======sortedYahoo======")
    for item in sortedYahoo:
        print(item[0])

def test():
    docs, classes = loadDataSet()
    vocabulary = createVocabList(docs)
    print("====test Vocabulary====")
    print("vocabulary", vocabulary)
    print(setOfWords2Vec(vocabulary, docs[0]))
    print("====test _trainNB0====")
    trainMatrix = []
    for doc in docs:
        trainMatrix.append(setOfWords2Vec(vocabulary, doc))
    p0V, p1V, pAb = _trainNB0(trainMatrix, classes)
    print("p0V", p0V)
    print("p1V", p1V)
    print("pAb", pAb)
    print("======test NB(trainNB0)======")
    testNB()
    print("======test Spam======")
    spamTest()
    print("======test localWords======")
    import feedparser
    # 这两个源不可用
    # ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    # sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    # nasa第0类，yahoo第一类
    nasa = feedparser.parse("http://www.nasa.gov/rss/dyn/image_of_the_day.rss")
    yahoo = feedparser.parse("http://sports.yahoo.com/nba/teams/hou/rss.xml")
    # localWords(nasa, yahoo)
    getTopWords(nasa, yahoo)


if __name__ == '__main__':
    test()
