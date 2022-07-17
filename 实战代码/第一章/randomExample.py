# load library
import numpy as np

# 设置种子
np.random.seed(0)

# 生成大小为3的随机数组
print(np.random.random(3))
# 生成3个在 1 和 10的随机整数
print(np.random.randint(0, 11, 3))
# 从均值为0的正态分布生成三个随机数
# 方差为1
print(np.random.normal(0.0, 1.0, 3))
# 从logistic分布中获得3个随机数
print(np.random.logistic(0.0, 1.0, 3))
# 从均值分布中获得3个随机数
print(np.random.uniform(1.0, 2.0, 3))