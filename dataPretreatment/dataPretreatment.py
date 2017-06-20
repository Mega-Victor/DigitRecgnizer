# !/usr/bin/python 3.5
# -*-coding:utf-8-*-

from numpy import *


# 把数据统一转换为int类型
def StrtoInt(data):
    data = mat(data)  # 转为矩阵处理
    m, n = shape(data)  # 维度大小
    newMatrix = zeros((m, n))  # 生成m*n大小的0矩阵存放转换后的数据
    for i in range(m):
        for j in range(n):
            newMatrix[i, j] = int(data[i, j])  # 把str转换为int类型
    return newMatrix


# 把非0的数据转化为1方便数据处理
def Normalized(data):
    m, n = shape(data)  # 获取数据维度
    for i in range(m):
        for j in range(n):
            # 数据为非0的情况下转为1
            if data[i, j] != 0:
                data[i, j] = 1
    return data
