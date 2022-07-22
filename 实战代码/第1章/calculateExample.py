# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# mean是算术平均值
print(np.mean(matrix))
# var是 方差
print(np.var(matrix))
# deviation 是标准差
print(np.std(matrix))

print(np.mean(matrix, axis=0))