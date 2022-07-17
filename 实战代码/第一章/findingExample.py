# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 最大值
print(np.max(matrix))
# 最小值
print(np.min(matrix))
# 每列最值
print(np.max(matrix, axis=0))
# 每行最值
print(np.max(matrix, axis=1))