#!usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#Logit模型 Function
#Logit（Logit model，也译作“评定模型”，“分类评定模型”，又作Logistic regression，“逻辑回归”）
def logit(x):
    return np.exp(x) / (1+np.exp(x))
#从-6到6取50个点，作为x轴的数
x=np.linspace(-6, 6, 50, dtype=float)
y=logit(x)
plt.plot(x,y)
plt.ylabel("Probability")
plt.show()