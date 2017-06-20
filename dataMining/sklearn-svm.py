#!/usr/bin/python 3.5
# -*- coding: utf-8 -*-

from numpy import *
from dataLoading.dataLoading import loadingTrainData, loadingTestData
from sklearn import svm
from dataSaving.dataSaving import dataSaving
import time


def sklearn_svm(trainData, trainLabel, testData):
    svcSvm = svm.SVC(C=5.0)
    svcSvm.fit(trainData, ravel(trainLabel))
    testLabel = svcSvm.predict(testData)
    print(testLabel)
    dataSaving(testLabel, 'F:\XianSheng\Projects\PythonProjects\DigitRecognizer\sklearn-svm.csv')
    return testLabel


start = time.clock()

trainData, trainLabel = loadingTrainData()
testData = loadingTestData()
result = sklearn_svm(trainData, trainLabel, testData)

elapsed = (time.clock() - start)
print("Time used:", int(elapsed/60), "min")
