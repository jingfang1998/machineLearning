#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
#这个数据文件中的第一行写明了列名，所以在这里不用指定columns了
admissions=pd.read_csv("admissions.csv")
#如果head里面不带参数，则代表显示前5个
print (admissions.head())

#画图的时候记得写上参数分别为横轴和纵轴代表什么
plt.scatter(admissions["gpa"], admissions["admit"], c="red")
plt.show()

