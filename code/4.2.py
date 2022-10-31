# 数据预处理
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

os.makedirs(os.path.join('/Users/zhanghanwen/Desktop/dive in ml', 'data'), exist_ok=True)
data_file = os.path.join('/Users/zhanghanwen/Desktop/dive in ml', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
# print(data)
# 处理缺失数据
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)  # 不是数值的可以分类
inputs = inputs.fillna(inputs.mean())  # 数值情况可以取均值
print(inputs)
x, y = torch.tensor(inputs.values, dtype=torch.float32), torch.tensor(outputs.values)
print(x, y)
