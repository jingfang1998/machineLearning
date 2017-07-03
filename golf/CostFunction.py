#!/usr/bin/python
# -*- coding:UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#通过distance来预测accuracy（准确率）

#先把数据做一个标准化，做一个平均值的归一化
pga=pd.read_csv("pga.csv")
pga.distance=(pga.distance-pga.distance.mean()) / pga.distance.std()
pga.accuracy=(pga.accuracy-pga.accuracy.mean()) / pga.accuracy.std()
print pga.head()

plt.scatter(pga.distance, pga.accuracy)
plt.xlabel("normalized distance")
plt.ylabel("normalized accuracy")
plt.show()

print 'shape of the series:', pga.distance.shape
print 'shape with newaxis:', pga.distance[:, np.newaxis].shape

logistic_model=LinearRegression()
#输入的模型的值写成以下sklearn库的标准形式
logistic_model.fit(pga.distance[:, np.newaxis], pga.accuracy)
# coef是一个通用的函数提取返回的对象建模功能模型系数
theta1=logistic_model.coef_[0]
print theta1

def cost(theta0, theta1, x, y):
    J=0
    m=len(x)
    for i in range(m):
        h=theta1 *x[i]+theta0
        J+=(h-y[i])**2
    J/=(2*m)
    return J

print cost(0, 1, pga.distance, pga.accuracy)

theta0=100
theta1s=np.linspace(-3, 2, 100)
costs=[]
for theta1 in theta1s:
    costs.append(cost(theta1, theta0, pga.distance, pga.accuracy))
plt.plot(theta1s, costs)
plt.show()




