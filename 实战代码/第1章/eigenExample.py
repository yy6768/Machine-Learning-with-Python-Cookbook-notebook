# load library
import numpy as np

# create matrix
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

#计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# 特征值
print(eigenvalues)
# 特征向量
print(eigenvectors)