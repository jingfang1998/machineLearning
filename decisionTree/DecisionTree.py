#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import math
import numpy as np
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "high_income"]
income = pd.read_csv("income.csv", names=columns)
# print(income.head(5))

#对数据进行预处理，把用字符串表示数据的列，都转化成数字表示
for name in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income"]:
    col=pd.Categorical.from_array(income[name])
    income[name]=col.codes

def calc_entropy(column):
    #counts得到一个数组（从大到小的数据）的个数
    counts=np.bincount(column)
    probabilties=counts / float(len(column))
    entropy = 0
    for proba in probabilties:
        entropy += proba * math.log(float(proba), 2)
    return -entropy

# print income.head()
#计算一下原始的信息熵
income_entropy=calc_entropy(income["high_income"])
print income_entropy

#选择按哪一列切分，然后计算信息增益
def calc_information_gain(data, split_name, target_name):
    #计算原始的信息熵
    origin_entropy=calc_entropy(data[target_name])
    #计算要切分的那一列的中位数，然后按中位数分为左右两个集合，左边的属于一类，右边的属于一类
    median = data[split_name].median()
    left_split = data[data[split_name] <= median]
    right_split = data[data[split_name] > median]

    # 循环 splits, and 计算subset entropy.
    now_entropy=0
    for subset in [left_split, right_split]:
        proba = subset.shape[0] / float(data.shape[0])
        now_entropy += proba * calc_entropy(subset[target_name])

    imfomation_gain=origin_entropy - now_entropy
    return imfomation_gain

print calc_information_gain(income, "age", "high_income")


#循环计算切分income中的每一列的信息增益,并找到信息增益最大的一列
def find_best_column(data, target_name, columns):
    information_gains = []
    # Loop through and compute information gains.
    for col in columns:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)
    # Find the name of the column with the highest gain.
    # 找到信息增益最大的那一列(这里显示第三列的增益最大)
    highest_gain_index = information_gains.index(max(information_gains))
    # 将这列定位到columns中的某一列（第3列），并返回第三列的名称
    highest_gain = columns[highest_gain_index]
    return highest_gain

fcolumns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
highest_gain = find_best_column(income, "high_income", fcolumns)
print highest_gain
#以上就完成了选择根节点的操作
#要完成选择根节点的操作，选择各列中信息增益最大的作为根节点
#对于信息增益最大一列进行切分，


data=pd.read_csv("data.csv")
# Create a dictionary to hold the tree.  This has to be outside the function so we can access it later.
tree = {}
# This list will let us number the nodes.  It has to be a list so we can access it inside the function.
nodes = []
#下面构建决策树，见note
