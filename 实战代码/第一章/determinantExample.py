# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 计算行列式
print(np.linalg.det(matrix))