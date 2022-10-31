# 手写线性回归
import os
import pandas as pd
import random

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from d2l import torch as d2l


# 随机生成数据集
def synthetic_data(w, b, num_examples):
    """"生成 y = xw + b + 噪声"""
    x = torch.normal(0, 1, (num_examples, len(w)))  # num_examples * len(w)大小
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4], dtype=torch.float32)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 可视化
# print('features:', features[0], '\nlabel:', labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

# 生成大小为batch_size的小批量，一个dataloader
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indexes = list(range(num_examples))
    # 随机打乱
    random.shuffle(indexes)
    for i in range(0, num_examples, batch_size):
        batch_indexes = torch.tensor(indexes[i:min(i + batch_size, num_examples)])
        yield features[batch_indexes], labels[batch_indexes]


# 可视化
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    """"线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法(简单随机梯度下降)
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 下一次计算梯度不会和上一次相关


batch_size = 20
lr = 0.01
num_epochs = 10
# net = linreg(X, w, b)
# loss =squared_loss()
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss = squared_loss(linreg(X, w, b), y)
        loss.sum().backward()
        sgd([w, b], lr, batch_size)
    # with torch.no_grad():表示当前计算不需要反向传播
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
