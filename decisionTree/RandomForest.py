#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "high_income"]
income = pd.read_csv("income.csv", names = columns)

#把字符串都变成整型数字
for name in ["education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income","workclass"]:
    col = pd.Categorical.from_array(income[name])
    income[name] = col.codes
#把样本的顺序打乱
shuffled_index = np.random.permutation(income.index)
income = income.reindex(shuffled_index)
#构造训练集和测试集
train_max_row = math.floor(income.shape[0] * .8)#math.floor()对浮点数向下取整
train = income.iloc[:int(train_max_row)]#将数据切片
test = income.iloc[int(train_max_row):]

#随机森林
#第一个参数是，设置几棵树，默认是10棵
features = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
rfc = RandomForestClassifier(n_estimators=5, random_state=1, min_samples_leaf=2)
rfc.fit(train[features], train["high_income"])
predictions = rfc.predict(test[features])
roc_auc_area=roc_auc_score(test["high_income"], predictions)
print roc_auc_area
