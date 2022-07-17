import numpy as np

# create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# select the third element of vector
# print(vector[2])
# print(matrix[1,1])
# print(vector[:])

#访问全部元素
print(vector[:])
# 切片访问
print(vector[:3])
# 逆向访问
print(vector[-1])
# 访问前两行
print(matrix[:2, :])
# 访问所有行，第二列
print(matrix[:,1:2])