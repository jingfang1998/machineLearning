#!/usr/bin/python
# -*- coding:UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#先把数据做一个标准化，做一个平均值的归一化
pga=pd.read_csv("pga.csv")
pga.distance=(pga.distance-pga.distance.mean()) / pga.distance.std()
pga.accuracy=(pga.accuracy-pga.accuracy.mean()) / pga.accuracy.std()
def cost(theta0, theta1, x, y):
    J=0
    m=len(x)
    for i in range(m):
        h=theta1 *x[i]+theta0
        J+=(h-y[i])**2
    J/=(2*m)
    return J

#为J（误差函数）对theta1求偏导
def partial_cost_thetal(theta0, theta1, x, y):
    h=theta1 * x + theta0
    diff=(h - y) * x
    partial=diff.sum() / (x.shape[0])
    return partial

def partial_cost_theta0(theta0, theta1, x, y):
    h=theta1 * x + theta0
    diff=h - y
    partial=diff.sum() / (x.shape[0])
    return partial

#theta1代表斜率，算一次cost，然后求偏导（得出的是斜率）让theta1减小偏导的这个值，直到cost值收敛
def gradient_descent(x, y, alpha=0.1, theta0=0, theta1=0):
    max_epochs=1000 #最大的循环次数
    counter=0
    c=cost(theta0, theta1, pga.distance, pga.accuracy)
    costs=[c]
    convergence_thres=0.000001 #两次的cost差值小于这个数，认为收敛
    cprev=c+10
    theta0s=[theta0]
    theta1s=[theta1]
    while((np.abs(cprev-c)>convergence_thres) and (counter<max_epochs)):
        cprev=c
        update0=alpha * partial_cost_theta0(theta0, theta1, x, y)
        update1=alpha * partial_cost_thetal(theta0, theta1, x, y)
        theta0 -= update0
        theta1 -= update1
        theta0s.append(theta0)
        theta1s.append(theta1)
        c=cost(theta0, theta1, pga.distance, pga.accuracy)
        costs.append(c)
        counter += 1
    return {"theta0": theta0, "theta1": theta1, "costs": costs}

theta0=gradient_descent(pga.distance, pga.accuracy)["theta0"]
theta1=gradient_descent(pga.distance, pga.accuracy)["theta1"]
print "theta0:",theta0
print "theta1:",theta1
descent=gradient_descent(pga.distance, pga.accuracy, alpha=.01)
plt.scatter(range(len(descent["costs"])), descent["costs"])
plt.show()