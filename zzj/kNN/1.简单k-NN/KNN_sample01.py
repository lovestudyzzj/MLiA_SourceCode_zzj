#-*- coding: utf-8 -*-
# @Time    : 2020-06-21 22:36
# @Author  : zongjizhu
# @File    : KNN_sample01.py
import numpy as np
import operator
import collections

"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2020-06-21
"""

def createDataSet():
    #四组二维特征
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    #四组特征的标签
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2020-06-21
"""
def classify0(inX, dataSet, labels, k):
	# 计算距离
	dist = np.sum((inX - dataSet)**2, axis=1)**0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

"""
函数说明:main函数

Parameters:
	无
Returns:
	无

Modify:
	2020-06-21
"""
if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [0,0]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)