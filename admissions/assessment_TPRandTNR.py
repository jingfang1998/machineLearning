#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
admissions=pd.read_csv("admissions.csv")
logistic_model=LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
predict_label=logistic_model.predict(admissions[["gpa"]])
admissions["predict_label"]=predict_label
admissions["actual_label"]=admissions["admit"]

#计算TPR（应该检测出来的正类的概率）
true_positive_filter=(admissions["actual_label"]==1) & (admissions["predict_label"]==1)
true_positive=admissions[true_positive_filter]
false_negative_filter=(admissions["actual_label"]==1) & (admissions["predict_label"]==0)
false_negative=admissions[false_negative_filter]
TPR=len(true_positive) / float(len(true_positive)+len(false_negative))
print 'TPR:',TPR

#计算TNR（应该不检测出来的负类，的概率）
true_negative_filter=(admissions["actual_label"]==0) & (admissions["predict_label"]==0)
true_negative=admissions[true_negative_filter]
false_positive_filter=(admissions["actual_label"]==0) & (admissions["predict_label"]==1)
false_positive=admissions[false_positive_filter]
TNR=len(true_negative) / float(len(true_negative)+len(false_positive))
print 'TNR:',TNR