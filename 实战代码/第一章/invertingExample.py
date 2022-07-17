# load library
import numpy as np

# create matrix
matrix = np.array([[1, 4],
                  [2, 5]])

# inv求逆
print(np.linalg.inv(matrix))
print(matrix @ np.linalg.inv(matrix))