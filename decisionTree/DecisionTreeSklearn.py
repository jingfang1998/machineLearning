#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import math
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "high_income"]
income = pd.read_csv("income.csv", names=columns)
#把字符串都变成整型数字
for name in ["education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income","workclass"]:
    col = pd.Categorical.from_array(income[name])
    income[name] = col.codes
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Set a random seed so the shuffle is the same every time.
np.random.seed(10)

# Shuffle the rows.  This first permutes the index randomly using numpy.random.permutation.
# Then, it reindexes the dataframe with this.
# The net effect is to put the rows into random order.
shuffled_index=np.random.permutation(income.index)
income = income.reindex(shuffled_index)

train_max_row = math.floor(income.shape[0] * .8)
#print train_max_row
train = income.iloc[:int(train_max_row)]
test = income.iloc[int(train_max_row):]

# Instantiate the classifier.
# Set random_state to 1 to keep results consistent.
dtc = DecisionTreeClassifier(random_state=1)
dtc.fit(train[columns], train["high_income"])

#用测试集预测（结果：0.776094276094）
predictions = dtc.predict(test[columns])
error = roc_auc_score(test["high_income"], predictions)
print(error)

#用训练集进行预测 （结果：0.990909090909）
predictions = dtc.predict(train[columns])
print(roc_auc_score(train["high_income"], predictions))
print '\n'

#由以上看出，训练集的预测结果和测试集的预测结果相差比较多，说明很有可能出现了过拟合的现象

#下面不使用默认参数，对树的深度以及各个参数进行设置，防止出现过拟合现象
# Decision trees model from the last screen.
dtc = DecisionTreeClassifier(min_samples_split=5, random_state=10)

dtc.fit(train[columns], train["high_income"])
predictions = dtc.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = dtc.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)
print '\n'


clf = DecisionTreeClassifier(random_state=1, min_samples_split=13, max_depth=7)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)
print '\n'

np.random.seed(10)

#加上一个没有意义的噪声特征，结果很容易出现过拟合
# Generate a column with random numbers from 0 to 4.
income["noise"] = np.random.randint(4, size=income.shape[0])

# Adjust columns to include the noise column.
columns = ["noise", "age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Make new train and test sets.
train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[:int(train_max_row)]
test = income.iloc[int(train_max_row):]

# Initialize the classifier.
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)
print '\n'


#随机森林的第一重随机性：样本的随机性
# We'll build 10 trees
tree_count = 10

# Each "bag" will have 60% of the number of original rows.
bag_proportion = .6

predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement.
    # We set a random state to ensure we'll be able to replicate our results.
    # We set it to i instead of a fixed value so we don't get the same sample every loop.
    # That would make all of our trees the same.
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)

    # Fit a decision tree model to the "bag".
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
    clf.fit(bag[columns], bag["high_income"])

    # Using the model, make predictions on the test data.
    predictions.append(clf.predict_proba(test[columns])[:, 1])
combined = np.sum(predictions, axis=0) / 10
rounded = np.round(combined)

print(roc_auc_score(test["high_income"], rounded))



#随机森林的第二重随机性：特征的随机性
# We'll build 10 trees
tree_count = 10

# Each "bag" will have 60% of the number of original rows.
bag_proportion = .6

predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement.
    # We set a random state to ensure we'll be able to replicate our results.
    # We set it to i instead of a fixed value so we don't get the same sample every time.
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)

    # Fit a decision tree model to the "bag".
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
    clf.fit(bag[columns], bag["high_income"])

    # Using the model, make predictions on the test data.
    predictions.append(clf.predict_proba(test[columns])[:, 1])

combined = np.sum(predictions, axis=0) / 10
rounded = np.round(combined)

print(roc_auc_score(test["high_income"], rounded))
predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement.
    # We set a random state to ensure we'll be able to replicate our results.
    # We set it to i instead of a fixed value so we don't get the same sample every time.
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)

    # Fit a decision tree model to the "bag".
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2, splitter="random", max_features="auto")
    clf.fit(bag[columns], bag["high_income"])

    # Using the model, make predictions on the test data.
    predictions.append(clf.predict_proba(test[columns])[:, 1])

combined = np.sum(predictions, axis=0) / 10
rounded = np.round(combined)

print(roc_auc_score(test["high_income"], rounded))