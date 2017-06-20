# !/usr/bin/python 3.5
# -*-coding:utf-8-*-
from numpy import *
from dataLoading.dataLoading import loadingTrainData, loadingTestData
from sklearn.naive_bayes import MultinomialNB
from dataSaving.dataSaving import dataSaving
import time


def sklearn_bayes_MNB(trainData, trainLabel, testData):
    skMNB = MultinomialNB(alpha=0.1)
    skMNB.fit(trainData, ravel(trainLabel))
    testLabel = skMNB.predict(testData)
    dataSaving(testLabel, 'F:\XianSheng\Projects\PythonProjects\DigitRecognizer\sklearn-bayes-MNB.csv')
    return testLabel


start = time.clock()

trainData, trainLabel = loadingTrainData()
testData = loadingTestData()
result = sklearn_bayes_MNB(trainData, trainLabel, testData)

elapsed = (time.clock() - start)
print("Time used:", int(elapsed/60), "min")