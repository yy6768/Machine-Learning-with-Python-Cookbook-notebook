# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix)

# #创建一个函数
# add_1000 = lambda i: i + 1000
#
#
# # vectorized
# vectorized_add_1000 = np.vectorize(add_1000)
#
# # 适用该函数
# vectorized_add_1000(matrix)
#
# print(matrix)

#广播
print(matrix+1000)