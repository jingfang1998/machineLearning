#!/usr/bin/python
# -*- coding: UTF-8 -*-
#先把数据映射成表格，然后用散点图表示一下
#对数据进行预处理并进行数据清洗的库，主要做数据处理和分析的
import pandas as pd
#实现数据可视化的库，散点图，直线图，柱形图
import matplotlib.pyplot as plt

#数据的特征：mpg（每加仑能行驶多少公里），有几个汽缸，引擎里的一个指标，车的动力有多大，车重，加速度，哪年生产的，车的产地（0，1，2）
columns = ["mpg","cylinders","displacements","horsepower","weight","acceleration","model year","origin","car name"]
#auto_mpg.data是文件的路径，这里文件和代码在一个目录下，因此直接写文件名，但建议使用绝对路径
#delim_whitespace=True指的是文件中用空格分开,csv文件是用逗号分开的（read_csv）
cars=pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
#打印出来前五行数据
print(cars.head(5))

#作用新建绘画窗口,独立显示绘画的图片
fig = plt.figure()
#指定一个分图，一共两行，一列，从上到下，从左到右数第一个图
ax1=fig.add_subplot(2,1,1)
#指定一个分图，一共两行，一列，从上到下，从左到右数第二个图
ax2=fig.add_subplot(2,1,2)
#横轴weight，纵轴mpg，kind=‘scatter’是散点图，mpg指的是每加仑行驶多少公里
cars.plot("weight", "mpg", kind='scatter', ax=ax1)
#横轴acceleration，纵轴mpg，kind=‘scatter’是散点图，mpg指的是每加仑行驶多少公里
cars.plot("acceleration", "mpg", kind='scatter', ax=ax2)
plt.show()

