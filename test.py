# -*- coding: utf-8 -*-
# @Time : 2022/11/3 10:34
# @Author : nlp_zzu
# @File : test.py
# @File_Description : 测试文件

import torch

print(torch.__version__)

# 初始化一个空矩阵
x = torch.empty(5,3)
print(x)

# 初始化一个随机矩阵(5,3) 5行 3列
x = torch.rand(5,3)
print(x)
print(type(x))

# 初始化一个全零矩阵
x = torch.zeros(5,3,dtype=torch.long)
print(x)


