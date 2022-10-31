# 线性代数
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# y = x.clone()
# print(x, x * y)  # 哈达玛积
# print(x.sum())
# x_sum_axis0 = x.sum(axis=[0, 1])  # 纬度求和
# print(x, x_sum_axis0, x_sum_axis0.shape)
# print(x.mean(), x.sum() / x.numel())
# print(x.mean(axis=0), x.sum(axis=0) / x.shape[0])
# sum_x = x.mean(axis=1, keepdims=True)
# print(x / sum_x)
# 累加求和
# print(x.cumsum(axis=0))
a = torch.arange(20, dtype=torch.float32).reshape(5, 4)
b = torch.ones(4, 3)
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
# dot只能对一维张量做点积
# print(x, y, torch.dot(x, y), torch.sum(x * y))
# 矩阵乘法
# print(torch.mv(a, x), torch.mm(a, b))
# l2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# l1范数
print(torch.abs(u).sum())
# frobenius范数
print(torch.norm(torch.ones(4, 9)))