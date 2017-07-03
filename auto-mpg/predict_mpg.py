#!/usr/bin/python
# -*- coding: UTF-8 -*-
#先把数据映射成表格，然后用散点图表示一下
#对数据进行预处理并进行数据清洗的库，主要做数据处理和分析的
import pandas as pd
#实现数据可视化的库，散点图，直线图，柱形图
import matplotlib.pyplot as plt
import sklearn
#从sklearn的库里面引入线性回归的算法
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
columns = ["mpg","cylinders","displacements","horsepower","weight","acceleration","model year","origin","car name"]
cars=pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)

#把线性回归的模型赋值给lr
lr = LinearRegression(fit_intercept=True)
#使用线性回归模型训练数据，第一个参数是input，代表用weight预测mpg，第二个参数是output，mpg，代表输出的标签是mpg的值
#目的是想看重量这个指标对与mpg的影响，这里cars还可以取多个值，后边cars["mpg"]指的是label值，label值是mpg值（每加仑油能走多少公里）
#训练模型的目的是为了与促
lr.fit(cars[["weight"]],cars["mpg"])
#本例子中测试的样本也使用cars，这样便于预测值与真实值的比较，新来了一批样本数据，来预测一下mpg值
#训练数据几个特征，测试数据就有几个特征，必须对应，这里直接用训练数据当作测试数据
predictions=lr.predict(cars[["weight"]])
#打印预测值
print(predictions[0:5])
#打印真实值
print(cars["mpg"][0:5])
#可以看到下面的预测结果误差比较大，下面用图形来表示一下真实值和预测值的效果
#红色的点代表真实值，蓝色的点代表预测的线性回归模型
plt.scatter(cars["weight"], cars["mpg"], c="red")
plt.scatter(cars["weight"], predictions, c="blue")
plt.show()

#均方误差的函数
mse=mean_squared_error(cars["mpg"], predictions)
print(mse)
#对上面的误差值开根号
result=mse ** (0.5)
print(result)