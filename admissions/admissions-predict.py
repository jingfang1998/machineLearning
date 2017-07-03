#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
adminssions=pd.read_csv("admissions.csv")
logistic_model=LogisticRegression()
logistic_model.fit(adminssions[["gpa"]], adminssions["admit"])
#这个预测的步骤和线性回归有些不同
#预测某一个类别属于某一类的概率是多少，下面这个函数是预测一下这个值的概率
#逻辑回归把西格玛函数y轴上中值以上当作当作一类，中值以下当作一类
#这个预测的结果是把这些值都转换成归于某一类的概率
pre_probs=logistic_model.predict_proba(adminssions[["gpa"]])
print pre_probs[:,1]
# pre_probs[:,1]的意思就是认为通过的概率是多少
plt.scatter(adminssions["gpa"], pre_probs[:,1])
#画出图来可以认为是随着gpa的增加，通过的概率也增大
plt.show()

#下面预测的是predict，这个函数，直接把分值归类，不是0，就是1
fit_label=logistic_model.predict(adminssions[["gpa"]])
adminssions["predict_label"]=fit_label
#输出了有几个0，有几个1
print (adminssions["predict_label"].value_counts())
print(adminssions.head())
plt.scatter(adminssions["gpa"], fit_label)
plt.show()
