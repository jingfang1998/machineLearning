#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplo
columns = ["mpg","cylinders","displacements","horsepower","weight","acceleration","model year","origin","car name"]
cars=pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
print(cars.head(5))

fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
cars.plot("weight", "mpg", kind="scatter", ax=ax1)
cars.plot("acceleration","mpg", kind="scatter", ax=ax2)
plt.show()