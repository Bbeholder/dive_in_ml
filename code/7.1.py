# 自动求导
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

x = torch.arange(4.0)
x.requires_grad_(True)
# print(x.grad)
y = 2 * torch.dot(x, x)
y.backward()
# print(x.grad, x.grad == 4 * x)
# x.grad.zero_()
# y = x * x
# # 等价于y.backward(torch.ones(len(x)))
# y.sum().backward()
# print(x.grad)
# 将某些操作移到计算图之外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)


# 控制流仍可计算变量的梯度
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
