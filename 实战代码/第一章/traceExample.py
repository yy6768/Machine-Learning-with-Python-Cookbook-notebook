# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 矩阵的迹
print(matrix.trace())
print(sum(matrix.diagonal()))