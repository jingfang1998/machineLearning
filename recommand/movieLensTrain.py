# !/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import cross_validation as cv
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
def load_file():
    # header =["userId", "movieId", "rating", "timestamp"]
    df = pd.read_csv("data/ratings.csv", sep=",")
    num_user = df.userId.unique().shape[0] # shape得到的是（行，列）= (138493,)
    num_movie = df.movieId.unique().shape[0] # shape得到的是（行，列）= (26744,)
    print("all users are %d...all movies are %d" %(num_user, num_movie))

    # 将样本分为训练集和测试集,创建用户-产品矩阵
    train_data, test_data = cv.train_test_split(df, test_size=0.25)
    train_data_matrix = np.zeros((num_user, num_movie))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3] #矩阵的行号和列号分别是userId和movieId，值是评分

    test_data_matrix = np.zeros((num_user, num_movie))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    # 创建相似性矩阵
    user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
    item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")
    return (train_data_matrix, test_data_matrix, user_similarity, item_similarity)

def predict(rating, similar, type = "user"):
    if type == "user":
        mean_user_rating = rating.mean(axis = 1) # 对行求均值
        # mean_user_rating 是一维矩阵，data_matrix是二维矩阵，所以给mean_user_rating增加了一维然后进行相减
        rating_diff = rating - mean_user_rating[:,np.newaxis]
        pred = mean_user_rating[:,np.newaxis] + similar.dot(rating_diff)/np.array([np.abs(similar).sum(axis=1)]).T
        return pred
    elif type == "item":
        pred = rating.dot(similar)/np.array([np.abs(similar).sum(axis=1)])
        return pred

def rmse(prediction, test_data_matrix):
    prediction = prediction[test_data_matrix.nonzero()].flatten()
    test_data_matrix = test_data_matrix[test_data_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_data_matrix))


if __name__ == "__main__":
    train_data_matrix, test_data_matrix, user_similarity, item_similarity =load_file()
    # print("user_similarity...", user_similarity)
    # print("item_similarity...", item_similarity)
    user_prediction = predict(train_data_matrix, user_similarity, type = "user")
    item_prediction = predict(train_data_matrix, item_similarity, type = "item")


    print('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('Item based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
