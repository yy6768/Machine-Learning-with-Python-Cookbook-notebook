# load library
import numpy as np

# create matricies
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# +
print(np.add(matrix_a, matrix_b))
# +
print(matrix_a + matrix_b)
# -
print(np.subtract(matrix_a, matrix_b))
# -
print(matrix_a - matrix_b)