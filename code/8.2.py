# 调库线性回归
import os
import pandas as pd
import random

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

# x = torch.arange(4.0)
# print(x)
# x.requires_grad_(True)
# # print(x.grad)
# y = (x * x).sum()
# y.backward()
# # print(x.grad, x.grad == 4 * x)
# # x.grad.zero_()
# # y = x * x
# # # 等价于y.backward(torch.ones(len(x)))
# # y.sum().backward()
# # print(x.grad)
# # 将某些操作移到计算图之外
# # x.grad.zero_()
# print(x.grad)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))
# 模型
net = nn.Sequential(nn.Linear(2, 1))
# 参数(初始化权重和偏差)
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


