#usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
admissions=pd.read_csv("admissions.csv")
admissions["actual_label"]=admissions["admit"]
admissions=admissions.drop("admit", axis=1)

#第一个参数：样本的长度
#第二个参数：把样本分成几份
#第三个参数：是否打乱样本的顺序
#第四个参数：是否指定一个种子来做这个随机操作
kf=KFold(len(admissions), 5, shuffle=True, random_state=8)
lr=LogisticRegression()
#计算精度（只需要改变scoring）
accuracies=cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="accuracy", cv=kf)
print accuracies
average_accuracies=sum(accuracies) / float(len(accuracies))
print average_accuracies

#计算roc_auc，roc曲线的面积
roc_aucs=cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="roc_auc", cv=kf)
print roc_aucs
average_roc=sum(roc_aucs) / float(len(roc_aucs))
print average_roc