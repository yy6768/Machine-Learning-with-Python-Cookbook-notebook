# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 转置矩阵
print(matrix.T)

# 转置向量
print(np.array([1, 2, 3, 4, 5, 6]).T)
# 转置 行向量
print(np.array([[1, 2, 3, 4, 5, 6]]).T)