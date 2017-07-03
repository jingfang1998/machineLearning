#!/usr/bin/python
# -*- coding:UTF-8 -*-
# 数据集使用鸢尾花数据集做机器学习任务
# 第一列是花萼长度，第二列是花萼宽度，第三列花瓣长度，第五列花瓣宽度，4个特征
# 根据四个特征将鸢尾花分成三种类别
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris=pd.read_csv("iris.csv")
shuffle_index=np.random.permutation(iris.index)
iris=iris.reindex(shuffle_index)
# print iris.head()
print iris["species"].unique()
#为每个数值的特征画出直方图，因为最后一列species是字符型的值，所以直方图没有显示
iris.hist()
plt.show()

#下面做一个矩阵相乘的算法dot，下面y.T是求y的转置
x=np.asanyarray([[9, 5, 4]])
y=np.asanyarray([[-1, 2, 4]])
# y.T求的是y的转置
print np.dot(x,y.T)

# shape[0]是看有多少行，np.ones把ones这一列都初始化为1，这里相当于初始化wx+b里面的b
iris["ones"] = np.ones(iris.shape[0])
# print np.ones(iris.shape[0])
print iris.head()

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
print a1

# 下面要计算的是loss的值
# First observation's features and target
x0 = X[0]
y0 = y[0]

# Initialize parameters, we have 5 units and just 1 layer
theta_init = np.random.normal(0,0.01,size=(5,1))
def singlecost(X, y, theta):
    # Compute activation
    h = sigmoid_activation(X.T, theta)
    # Take the negative average of target*log(activation) + (1-target) * log(1-activation)
    cost = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

first_cost = singlecost(x0, y0, theta_init)
print 'first_cost:',first_cost

#下面主要是更新theta的操作
# 接下来要看的是theta1产生多大的loss，theta2产生多大loss，theta3产生多大的loss
# 通过变换theta的值，来使得loss的值最小，所以我们就看权重参数对loss做了多大贡献
# 我们给每个theta值求偏导
# Initialize parameters
theta_init = np.random.normal(0,0.01,size=(5,1))#5行1列个theta值
# Store the updates into this array
grads = np.zeros(theta_init.shape)#5行1列个0，初始化前进的方向
# Number of observations计算的是这些特征+ones一共有多少行
# X = iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
n = X.shape[0]
# enumerate(X)返回的是一个（序号，X），因此下边的j只是一个序号，obs代表X
for j, obs in enumerate(X):
    # Compute activation
    h = sigmoid_activation(obs, theta_init)
    # Get delta
    delta = (y[j]-h) * h * (1-h) * obs
    # accumulate，更新theta值
    # delta是[ 0.12355751  0.6795663   0.29653802  0.45716278  0.12355751]
    # delta[:,np.newaxis]是[[ 0.12355751] [ 0.6795663 ][ 0.29653802][ 0.45716278][ 0.12355751]]
    # 这里这样做是为了让delta和grads变成相同的形式
    # grads的作用是指明前进的方向
    grads += delta[:,np.newaxis]/X.shape[0]
    #print delta[:,np.newaxis]/X.shape[0]


#整合上边
theta_init = np.random.normal(0, 0.01, size=(5, 1))
# set a learning rate，每次沿grads的方向走多少不步
learning_rate = 0.1
# maximum number of iterations for gradient descent，最大的循环次数
maxepochs = 10000
# costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.0001
def learn(X, y, theta, learning_rate, maxepochs, convergence_thres):
    costs = []
    cost = singlecost(X, y, theta)  # compute initial cost
    costprev = cost + convergence_thres + 0.01  # set an inital costprev to past while loop
    counter = 0  # add a counter
    # Loop through until convergence
    for counter in range(maxepochs):
        grads = np.zeros(theta.shape)#先将这个前进的方向初始化为0
        for j, obs in enumerate(X):
            h = sigmoid_activation(obs, theta)  # Compute activation计算h
            #对theta（是一个向量，包括theta1，theta2）求偏导
            delta = (y[j] - h) * h * (1 - h) * obs  # Get delta
            grads += delta[:, np.newaxis] / X.shape[0]  # accumulate，对grads进行填充

        # update parameters
        theta += grads * learning_rate
        counter += 1  # count
        costprev = cost  # store prev cost
        cost = singlecost(X, y, theta)  # compute new cost
        costs.append(cost)
        if np.abs(costprev - cost) < convergence_thres:
            break

    plt.plot(costs)
    plt.title("Convergence of the Cost Function")
    plt.ylabel("J($\Theta$)")
    plt.xlabel("Iteration")
    plt.show()
    return theta

theta = learn(X, y, theta_init, learning_rate, maxepochs, convergence_thres)


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