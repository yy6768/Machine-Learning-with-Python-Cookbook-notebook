# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 对角线
print(matrix.diagonal())
print(matrix.diagonal(offset=1))
print(matrix.diagonal(offset=-1))