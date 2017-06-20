# !/usr/bin/python 3.5
# -*-coding:utf-8-*-
import csv


def dataSaving(result, filename):
    """
    
    :param result: 保存数据
    :param filename: 文件名 
    :return: 
    """
    with open(filename, 'w') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            l = []
            l.append(i)
            myWriter.writerow(l)
