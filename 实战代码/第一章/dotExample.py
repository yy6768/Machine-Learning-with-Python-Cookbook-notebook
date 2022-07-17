# load library
import numpy as np

# create two vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# 点积
print(np.dot(vector_a, vector_b))
print(vector_a@vector_b)