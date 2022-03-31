import torch as t
a = t.arange(0, 2304).view(128, 18)
print(a)
print(a.shape)
# 选取对角线的元素
index = t.arange(0, 128).view(128, 1)
print(a.gather(1, index))
print(index.shape)
