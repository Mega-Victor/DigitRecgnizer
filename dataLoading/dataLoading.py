# !/usr/bin/python 3.5
# -*-coding:utf-8-*-
from numpy import array
from dataPretreatment.dataPretreatment import Normalized, StrtoInt
import csv


# 加载训练集
def loadingTrainData():
    l = []
    with open('F:\XianSheng\Projects\PythonProjects\DigitRecognizer/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])  # 除去第一行标签名称这个数据
    l = array(l)
    label = l[:, 0]  # label列
    data = l[:, 1:]  # 数据列
    return Normalized(StrtoInt(data)), StrtoInt(label)  # label 1*42000  data 42000*784


# 加载测试集
def loadingTestData():
    l = []
    with open('F:\XianSheng\Projects\PythonProjects\DigitRecognizer/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
            # 28001*784
    l.remove(l[0])  # 去除第一行标称
    data = array(l)
    return Normalized(StrtoInt(data))  # data 28000*784
