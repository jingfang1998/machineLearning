# !/usr/bin/python
# -*- coding:UTF-8 -*-
# 数据是NBA球员的数据，球员信息统计指标
# 我们现在只用部分指标
# 数据有400多名NBA球员
# pos代表的是球员打的是什么位置（c=中锋），投篮命中率，罚球命中率，三分，助攻等一系列指标
# 我们的数据只是包含打球位置是'控球后卫'的球员，即pos='PG',只衡量打这个位置的球员
# pts这个指标指的是这个球员的打球的总得分，g表示的是这个球员一共打了多少场球
# ppg=pts/g指的是这个球员平均每场球能得多少分
# 评价一个控球后卫打球打的好坏，用助攻总数ast，还有失误总数tov
# atr=ast/tov来评价一个球员打球打的怎么样（大概的效率）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
nba=pd.read_csv("nba_2013.csv")
point_guards=nba[nba["pos"] == 'PG']
#=1的目的是防止数据中有0，除数不能为0
point_guards["ppg"]=point_guards["pts"] / (point_guards["g"]+1)
point_guards["atr"]=point_guards["ast"] / (point_guards["tov"]+1)
point_guards["cluster"]=0

#先画个散点图看这两个指标有什么效果
# plt.scatter(x=point_guards["ppg"], y=point_guards["atr"], c="yellow")
# plt.show()

num_cluster=5

def visualize_cluster(df, num_clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for n in range(num_clusters):
        clusters_df=df[df["cluster"] == n]
        plt.scatter(x=clusters_df["ppg"], y=clusters_df["atr"], c=colors[n-1])
    plt.xlabel('Points Per Game', fontsize=13)
    plt.ylabel('Assist Turnover Ratio', fontsize=13)
    plt.show()


kmeans_model = KMeans(n_clusters=num_cluster)
kmeans_model.fit(point_guards[['ppg', 'atr']])
point_guards["cluster"] = kmeans_model.labels_

visualize_cluster(point_guards, num_cluster)

