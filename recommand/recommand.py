# !/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
from math import sqrt
# 该dictionary以（行名，列名）为索引
def load_matrix():
    matrix = {}
    file = open("data/train.csv")
    columns = file.readline().split(",")
    for line in file:
        scores = line.split(",")
        name = scores[0]
        for i in range(1, len(scores)):
            matrix[(name, columns[i])] = scores[i].strip("\n")
    return matrix

def top_matches(matrix, sure_name):
    names = set(map(lambda l:l[0], matrix.keys()))
    scores = [(sim_distance(matrix, sure_name, name),name) for name in names if name!=sure_name]
    scores.sort(reverse=True)
    return scores

def sim_distance(matrix, sure_name, name):
    sure_name_scores = {sk[1]:matrix.get(sk) for sk in matrix if sk[0]== sure_name}
    name_score = {sk[1]:matrix.get(sk) for sk in matrix if sk[0]== name}
    ou_distance = sqrt(sum(pow(float(sure_name_scores.get(key))-float(name_score.get(key)), 2)for key in name_score.keys() if sure_name_scores.get(key)!="" and name_score.get(key)!=""))
    # 归一化
    p = 1/(1+ou_distance)
    return p
# 矩阵的转置
def transform_matrix(matrix):
    rows = set(map(lambda l:l[0], matrix.keys()))
    columns = set(map(lambda l:l[1], matrix.keys()))
    transform_matrix = {}
    for row in rows:
        for column in columns:
            transform_matrix[(column, row)] = matrix[(row, column)]
    return transform_matrix

# 找到某人感兴趣的影片
def get_recommendations(matrix, name):
    persons = set(map(lambda l:l[0], matrix.keys()))
    films = set(map(lambda l:l[1], matrix.keys()))
    recommand_value = {}
    sim = {}
    for person in persons:
        if person == name:
            continue
        person_sim = sim_distance(matrix, name, person)
        if person_sim <0:
            continue
        for film in films:
            if matrix[(person, film)] == "":
                continue
            sim.setdefault(film, 0)
            sim[film] += person_sim
            recommand_value.setdefault(film, 0)
            recommand_value[film] += float(matrix[(person, film)]) * person_sim
    score = [(recommand_value[film]/sim[film],film) for film in recommand_value]
    score.sort(reverse=True)
    return score



if __name__ == "__main__":
    #给某人推荐电影
    print(get_recommendations(load_matrix(), "Kai Zhou"))

    #找对某电影感兴趣的人
    print(get_recommendations(transform_matrix(load_matrix()), "Friends"))

    # 比较Kai Zhou和其他人的相似性
    # print(top_matches(load_matrix(), "Kai Zhou"))

    # 计算和Friends相似的影片
    # transform_matrix = transform_matrix(load_matrix())
    # print(top_matches(transform_matrix, "Friends"))