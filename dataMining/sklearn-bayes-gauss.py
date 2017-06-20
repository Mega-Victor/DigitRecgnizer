#!/usr/bin/python 3.5
# -*- coding: utf-8 -*-

from numpy import *
from dataLoading.dataLoading import loadingTrainData, loadingTestData
from sklearn.naive_bayes import GaussianNB
from dataSaving.dataSaving import dataSaving
import time

def sklearn_bayes_Gauss(trainData, trainLabel, testData):
    skGauss = GaussianNB()
    skGauss.fit(trainData, ravel(trainLabel))
    testLabel = skGauss.predict(testData)
    print(testLabel)
    dataSaving(testLabel, 'F:\XianSheng\Projects\PythonProjects\DigitRecognizer\sklearn_bayes-gauss.csv')
    return testLabel

start = time.clock()

trainData, trainLabel = loadingTrainData()
testData = loadingTestData()
result = sklearn_bayes_Gauss(trainData, trainLabel, testData)

elapsed = (time.clock() - start)
print("Time used:", int(elapsed/60), "min")
