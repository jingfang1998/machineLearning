#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
columns = ["mpg","cylinders","displacements","horsepower","weight","acceleration","model year","origin","car name"]
cars=pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
#把汽缸数量转化为多个列,prefix,指的是给转化后的各个列加一个前缀
dumny_cylinders=pd.get_dummies(cars["cylinders"], prefix="cyl")
cars=pd.concat([cars, dumny_cylinders], axis=1)
#print cars.head()

#把year也转化为多个列
dumny_year=pd.get_dummies(cars["model year"], prefix="year")
cars=pd.concat([cars, dumny_year], axis=1)
#print cars.head()

#将原来的两个特征删除
cars=cars.drop("model year", axis=1)
cars=cars.drop("cylinders", axis=1)
#print cars.head()

shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]

#origin是类别,筛选出来一共有多少类,然后排序
unique_origins=cars["origin"].unique()
unique_origins.sort()

models = {}
#选出用来训练模型的列，这里用刚刚处理过的列来训练数据
features=[c for c in train.columns if c.startswith("cyl") or c.startswith("year")]

#一共分成几类，就把模型训练几次，并把每次训练的模型都存起来
for origin in unique_origins:
    model=LogisticRegression()
    #通过features来训练这个模型，第二个参数指的是，如果是当前循环的这个origin就把它当成正类
    #其余的特征都认为成负类
    # 转化为多个二分类问题，把ABCD都分别当成正类，其他都当成负类
    # A - 正，其他负，得出是A的概率
    # B - 正，其他负，得出是B的概率
    # C - 正，其他负，得出是C的概率

    model.fit(train[features], train["origin"] == origin)
    models[origin]=model

#开始预测
#首先构造了一个框架，以各个origin为列
testing_proba=pd.DataFrame(columns=unique_origins)
print testing_proba
for origin in unique_origins:
    #对于单分类来说，预测的结果是，features的每个值对应的一个是1的概率
    #但是这个是多分类，而且features里面所有的列可以综合起来看成一个列
    #得出的结果是，这一个列有很多值，都对应一个是1的概率，竖着看就好
    #在这里是有三个分类，因此有三列
    testing_proba[origin] = models[origin].predict_proba(test[features])[:, 1]
print testing_proba

#axis=1代表行
predict_origin=testing_proba.idxmax(axis=1)
print predict_origin