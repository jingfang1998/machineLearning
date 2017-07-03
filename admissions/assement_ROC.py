#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
admissions=pd.read_csv("admissions.csv")
admissions["actual_label"]=admissions["admit"]
admissions.drop("admit", axis=1)
np.random.seed(8)
shuffled_index=np.random.permutation(admissions.index)
shuffled_admissions=admissions.loc[shuffled_index]
train=shuffled_admissions.iloc[0:515]
test=shuffled_admissions.iloc[515:len(shuffled_admissions)]
logistic_model=LogisticRegression()
logistic_model.fit(train[["gpa"]], train["actual_label"])
probabilities=logistic_model.predict_proba(test[["gpa"]])
#fpr和tpr是两个指标，代表对于正类预测的准确率和对于负类预测的准确率，threshoders代表取的各个域值
#输入是真实的label，还有取各个gpa值所对应的分类为1的概率和分类为0的概率
fpr, tpr, thresholds=metrics.roc_curve(test["actual_label"], probabilities[:,1])
#print thresholds
plt.plot(fpr, tpr)
#对于正例的预测的准确率越高越好，对于负类的预测的准确率也是越高越好，所以，曲线的面积趋于1，模型效果好
plt.show()
#计算ROC曲线的面积
auc_score=roc_auc_score(test["actual_label"], probabilities[:,1])
print 'auc_score:',auc_score