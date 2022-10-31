# 数据操作实现
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

x = torch.arange(12)
print(x.shape)  # 张量形状
# print(x)  # 张量
# print(x.numel())  # 张量元素总数
# 改变一个张量的形状而不改变其元素和元素值
# X = x.reshape(3, 4)
# print(X)
# 全零全一
# a = torch.zeros((2, 3, 4))  # 2dimension,3row,4column
# b = torch.ones((2, 3, 4))
# print(a, b)
# 自定义张量
# x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 张量运算
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print(x + y, x - y, x * y, x / y, x ** y)
# print(torch.exp(x))
# 多个张量连结
# x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 添加记录和拼接特征
# print(torch.cat((x, y), dim=0), torch.cat((x, y), dim=1))
# 逻辑与求和
# print(x == y)
# print(x.sum())
# 广播机制
# x = torch.arange(3).reshape((3, 1))
# y = torch.arange(3).reshape((3, 1))
# print(x + y)
# 访问最后一行
# print(x[-1])
# 访问第一行和第二行
# print(x[1:3])
# 单行多行赋值
# x[1, 0] = 9
# x[0:2, :] = 12
# before = id(y)
# y = y + x
# print(id(y) == before)
# 原地操作
# z = torch.zeros_like(y)
# print('id(z):', id(z))
# z[:] = x + y
# print('id(z):', id(z))
# before = id(x)
# x += y
# print(id(x) == before)
# print(x)
# print(y)
# 转换为numpy张量
# x = torch.arange(3)
# y = torch.tensor([3.5])
# a = x.numpy()
# b = torch.tensor(a)
# print(type(a), type(b))
# # 将大小为1的张量转换为python标量
# print(y, y.item(), float(y), int(y))
