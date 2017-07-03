#!usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
columns = ["mpg","cylinders","displacements","horsepower","weight","acceleration","model year","origin","car name"]
cars=pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
lr=LinearRegression(fit_intercept=True)
lr.fit(cars[["weight"]], cars["mpg"])
predictions=lr.predict(cars[["weight"]])
print (predictions[0:5])
print (cars["mpg"][0:5])

plt.scatter(cars["weight"], cars["mpg"], c="red")
plt.scatter(cars["weight"], predictions, c="blue")
plt.show()

mse=mean_squared_error(cars["mpg"], predictions)
print (mse)
result=mse ** (0.5)
print (result)
