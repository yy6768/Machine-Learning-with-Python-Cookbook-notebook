# load library
import numpy as np

# create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# 重构
print(matrix.reshape(2, 6))

# 大小
print(matrix.size)
# -1表示京可能多的列
print(matrix.reshape(1, -1))

#压缩成一维
print(matrix.reshape(12))