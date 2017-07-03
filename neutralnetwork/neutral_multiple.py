#!/usr/bin/python
# -*- coding:UTF-8 -*-
# 数据集使用鸢尾花数据集做机器学习任务
# 第一列是花萼长度，第二列是花萼宽度，第三列花瓣长度，第五列花瓣宽度，4个特征
# 根据四个特征将鸢尾花分成三种类别
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
iris=pd.read_csv("iris.csv")
shuffle_index=np.random.permutation(iris.index)
iris=iris.reindex(shuffle_index)


# shape[0]是看有多少行，np.ones把ones这一列都初始化为1，这里相当于初始化wx+b里面的b
iris["ones"] = np.ones(iris.shape[0])

#下面的X指的是输入的四个特征x，y指的是最后输出的结果（这里我们只判断一个二分类的状况）
# species=Iris-versicolor就是1这个类别，不是Iris-versicolor这个种类就是0这个类别
X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = (iris.species == 'Iris-versicolor').values.astype(int)

# The first observation取一行样本进行测试
x0 = X[0]

# Initialize thetas randomly（size=(5, 1)指的是5行1列）
theta_init = np.random.normal(0, 0.01, size=(5, 1))

#输入theta和x进行h(x)的计算
def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))

a1 = sigmoid_activation(x0, theta_init)


# Use a class for this model, it's good practice and condenses the code
class NNet3:
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

    def _multiplecost(self, X, y):
        # feed through network
        l1, l2 = self._feedforward(X)
        # compute error
        inner = y * np.log(l2) + (1 - y) * np.log(1 - l2)
        # negative of average error
        return -np.mean(inner)

    def _feedforward(self, X):
        # feedforward to the first layer
        l1 = sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2

    def predict(self, X):
        _, y = self._feedforward(X)
        return y

    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0, 0.01, size=(ncols, self.hidden_layer))
        self.theta1 = np.random.normal(0, 0.01, size=(self.hidden_layer + 1, 1))

        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres + 1  # set an inital costprev to past while loop
        counter = 0  # intialize a counter

        # Loop through until convergence
        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self._feedforward(X)

            # Start Backpropagation
            # Compute gradients
            l2_delta = (y - l2) * l2 * (1 - l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1 - l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:, 1:] / nobs * self.learning_rate

            # Store costs and check for convergence
            counter += 1  # Count
            costprev = cost  # Store prev cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev - cost) < self.convergence_thres and counter > 500:
                break

# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001
# Number of hidden units
hidden_units = 4

# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model
model.learn(X, y)

# Plot costs
plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()

# First 70 rows to X_train and y_train
# Last 30 rows to X_train and y_train
X_train = X[:70]
y_train = y[:70]

X_test = X[-30:]
y_test = y[-30:]

# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001
# Number of hidden units
hidden_units = 4

# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
model.learn(X_train, y_train)

yhat = model.predict(X_test)[0]

auc = roc_auc_score(y_test, yhat)
print auc