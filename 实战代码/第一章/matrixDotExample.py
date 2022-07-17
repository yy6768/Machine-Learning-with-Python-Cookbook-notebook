# load library
import numpy as np

# create matrices
matrix_a = np.array([[1, 1],
                     [1, 2]])

matrix_b = np.array([[1, 3],
                     [1, 2]])

# 点积
print(np.dot(matrix_a, matrix_b))
# @
print(matrix_a @ matrix_b)