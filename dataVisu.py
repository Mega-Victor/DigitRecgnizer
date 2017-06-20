# !/usr/bin/python 3.5
# -*-coding:utf-8-*-
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
# 使用PCA把数据降为3维，寻找高维空间中，
# 数据变化最快（方差最大）的方向，对空间的基进行变换，
# 然后选取重要的空间基来对数据降维，以尽可能的保持数据特征的情况下对数据进行降维
pca = decomposition.PCA(n_components=3)
# 训练算法，设置内部参数，数据转换。也就是合并fit和transform方法
new_X = pca.fit_transform(X)
# 使用Axes3D进行3D绘图
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.spectral)
plt.show()
