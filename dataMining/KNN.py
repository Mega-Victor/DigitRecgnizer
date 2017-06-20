# !/usr/bin/python 3.5
# -*-coding:utf-8-*-
from numpy import *
import pandas as pd
import operator
import time
from dataLoading.dataLoading import loadingTestData,loadingTrainData
from dataSaving.dataSaving import dataSaving


def KNN(data, dataSet, labels, k):
    """
    k-NearestNeighbor从训练样本集中选择k个与测试样本“距离”最近的样本，
    这k个样本中出现频率最高的类别即作为测试样本的类别
    :param data:  除去标签之外的测试向量 1*n
    :param dataSet: 整个数据集 m*n，一行是一个样本
    :param labels: 标签集 m*1
    :param k: KNN的k值
    :return: 
    """
    data = mat(data)
    dataSet = mat(dataSet)
    labels = mat(labels)  # 数据全部矩阵化
    dataSetSize = dataSet.shape[0] # 得到dataSet的行数也就是样本个数
    diffMat = tile(data, (dataSetSize, 1)) - dataSet # tile(A,(m,n))  将数组A作为元素构造出m行n列的数组
    sqDiffMat = array(diffMat) ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # array.sum(axis=1)按行累加，axis=0为按列累加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # array.argsort()，得到每个元素的排序序号
    classCount = {}  # sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i], 0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key,x)从字典中获取key对应的value，没有key的话返回0
    # sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def KNN_DigitRecognizer():
    trainData, trainLabel = loadingTrainData()
    testData = loadingTestData()
    m, n = shape(testData)
    resultList = []
    for i in range(m):
        knnResult = KNN(testData[i], trainData, trainLabel.transpose(), 5)  # 需要转置处理
        print(knnResult)
        resultList.append(knnResult)
     # # end time
    dataSaving(resultList,'F:\XianSheng\Projects\PythonProjects\DigitRecognizer\knn.csv')

start = time.clock()
KNN_DigitRecognizer()
elapsed = (time.clock() - start)
print("Time used:", int(elapsed/60), "min")


