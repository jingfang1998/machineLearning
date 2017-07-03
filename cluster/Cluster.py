#!/usr/bin/python
# -*- coding:UTF-8 -*-
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
votes=pd.read_csv("114_congress.csv")
counts=votes["party"].value_counts()
# print counts
average=np.mean(votes, axis=0)#求均值（数字列的均值）
# print average

#衡量两个人投票的方式是否一样
#欧式距离在sklearn这个库里面有（第一个参数是第一个样本，第二个参数是第二个样本）
#iloc[0,3:]指的是0号样本，从第3列开始计算（第三列开始才是投票情况）
#reshape(1,-1)是把样本转化成一个行向量的形式，通过行向量做一个相似度对比
#这里是二维数组，指定了一维，二维可以自己推断，这里的-1表示自己推断
#这里相当于把矩阵的两行转化成了两个行向量，然后计算两个行向量之间的欧式距离
distance=euclidean_distances(votes.iloc[0,3:].reshape(1, -1), votes.iloc[1,3:].reshape(1, -1))
print distance
#votes.iloc[0,3:]里面的0代表第0行，3:代表列是从第三列开始的
#reshape(1, -1)代表把这一行转化为行向量，-1代表列的情况自己判断
print votes.iloc[0,3:].reshape(1, -1)#把votes矩阵中的第0行和第三列以后的数据转化为行向量
print votes.iloc[1,3:].reshape(1, -1)#把votes矩阵中的第1行和第三列以后的数据转化为行向量


#kmeans算法的使用
#random_state为1使得每次聚类的结果都是一样的，如果不为1，每次运行都会发生一些改变
#n_clusters指的是要把这些样本分成几类
kmeans_model=KMeans(n_clusters=2, random_state=1)
#将原始的样本分成了两簇，senator_distance结果中
# 显示的是每个样本（每行数据）距离每个簇（这里有两个簇）的距离
# 下面的变量的英文是参议员的意思
senator_distance=kmeans_model.fit_transform(votes.iloc[:,3:])
print senator_distance

#下面这个label是模型给各个样本分成了两类（样本只有0和1）
#如果上边的n_clusters=3，则下边的label的种类会变成（0，1，2）
labels = kmeans_model.labels_
# print labels
#下边这条命令，首先显示了party中有几种值，然后以上的label值在这几种值之间的分布情况
#显示了对于party中的三个种类在两个簇中的分布情况
print pd.crosstab(labels, votes["party"])
#我们之前的假设是，同一个党派的投票形式相似，会聚成一类（这里有共和党和民主党）
#上述的结果证明了我们的假设

#下面的这个方法输出的是（labels=1而且party不是D的那些样本）
#下面的按位与操作，只有两边同时满足条件，才会成立
republican = votes[(labels == 1) & (votes["party"] != "D")]
print republican.shape[0]

#上边我们求过每个样本对于各个簇的距离
# 下面我们把样本对于第一个簇的距离作为x轴，把样本对于第二个簇的距离作为y轴，画出散点图
# c指的是color和label保持一致，一部分是深蓝色，一部分是另外一种颜色
# 这个图中能看出来，两个簇的分布情况，从中间的一个点分开了，左边一簇，右边一簇
plt.scatter(x=senator_distance[:, 0], y=senator_distance[:, 1], c=labels)
plt.show()

#更多情况下我们是要找一个离群点，我们可以把边缘点当成离群点，，离群点的意思就是这个点，要么距离第一个簇的距离比较大，要么距离第二个簇的距离比较大，总之就是下面senator_distance矩阵中的每行的各项值比较大
# 我们在上边说过，senator_distance表示的是，每个样本距离各簇的距离
extremism=(senator_distance ** 3).sum(axis=1)
votes["extremism"]=extremism
#inplace=True，不创建新的对象，直接在原始对象上尽心修改
# ascending=False是不升序的意思
votes.sort_values("extremism", inplace=True, ascending=False)
print votes.head()