#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.linear_model import  LogisticRegression
admissions=pd.read_csv("admissions.csv")
logistic_model=LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
fit_label=logistic_model.predict(admissions[["gpa"]])
#将真实值，和预测值进行比较，并输出正确率
admissions["predict_label"]=fit_label
admissions["actual_label"]=admissions["admit"]
matches=admissions["actual_label"] == admissions["predict_label"]
#print (matches)
correct_predictions=admissions[matches]
#print (correct_predictions.head())
#输出正确率（精度），但是使用这种精度来评判，对于样本分布不均匀的情况下，是不负责任的
accuracy= len(correct_predictions) / float(len(admissions))
print 'accuracy:',accuracy
