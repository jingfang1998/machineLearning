#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
admissions=pd.read_csv("admissions.csv")
admissions["actual_label"]=admissions["admit"]
admissions.drop("admit", axis=1)
shuffled_index=np.random.permutation(admissions.index)
#把admissions打乱
shuffled_admissions=admissions.loc[shuffled_index]
#print shuffled_admissions.head()
#将admissions按照打乱的顺序，重新编号
admissions=shuffled_admissions.reset_index()
#print admissions.head()
admissions.ix[0:128, "fold"]=1
admissions.ix[129:257, "fold"]=2
admissions.ix[258:386, "fold"]=3
admissions.ix[387:514, "fold"]=4
admissions.ix[515:644, "fold"]=5
admissions["fold"]=admissions["fold"].astype("int")
print admissions.head()
print admissions.tail()
# train=admissions[admissions["fold"]!=1]
# test=admissions[admissions["fold"]==1]
# #training
# logistic_model=LogisticRegression()
# logistic_model.fit(train[["gpa"]], train["actual_label"])
# #predicting
# predict_label=logistic_model.predict(test[["gpa"]])
# test["predict_label"]=predict_label
#
# matches=test["actual_label"] == test["predict_label"]
# correct_label=test[matches]
# accuracy=len(correct_label) / float(len(test["predict_label"]))
# print 'accuracy:',accuracy

fold_ids=[1, 2, 3, 4, 5]
def train_and_test(df, folds):
    fold_accuracies=[]
    for fold in folds:
        logistic_model=LogisticRegression()
        train=admissions[admissions["fold"] != fold]
        test=admissions[admissions["fold"] == fold]
        logistic_model.fit(train[["gpa"]], train["actual_label"])
        predict_label=logistic_model.predict((test[["gpa"]]))
        test["predict_label"]=predict_label
        matches=test["actual_label"] == test["predict_label"]
        correct=test[matches]
        accuracy=len(correct) / float(len(test))
        fold_accuracies.append(accuracy)
    return fold_accuracies

accr=train_and_test(admissions, fold_ids)
print 'accuracies:', accr

average_accuracy=np.mean(accr)
print 'average_accuracy:', average_accuracy