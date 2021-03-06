# Machine Learning with Python Cookbook 学习笔记



### 前言

- 本笔记是针对人工智能典型算法的课程中Machine Learning with Python Cookbook的学习笔记
- 学习的实战代码都放在代码压缩包中
- 实战代码的运行环境是**python3.9   numpy 1.23.1 anaconda 4.12.0 **
- 上一章：
- 代码仓库
  - Github:[yy6768/Machine-Learning-with-Python-Cookbook-notebook: 人工智能典型算法笔记 (github.com)](https://github.com/yy6768/Machine-Learning-with-Python-Cookbook-notebook)
  - Gitee:[yyorange/机器学习笔记代码仓库 (gitee.com)](https://gitee.com/yyorange/Machine-Learning)


## Chapter 1 **Vectors, Matrices, and Arrays**

vector（向量）

matrice(矩阵)

array（数组）



### 1.0  简介

- numpy是python机器学习的基础
- 它可以对机器学习常用的数据结构进行操作
- 操作支持向量矩阵和数组

numpy相关资料：

[NumPy 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/numpy/numpy-tutorial.html)



### 1.1 创建一个vector

#### **Problem**

You need to create a vector

#### **Solution**

Use Numpy to create a one-dimensional array

在numpy中一维数组等价于向量

**numpy.array函数**

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

**vectorExample.py**

```python
# 引入numpy库
import numpy as np

# 创建行向量
vector_row = np.array([1, 2, 3])

# 创建列向量
vector_column = np.array([[1],
                          [2],
                          [3]])
```

![image-20220711203552886](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711203552886.png)

分别打印两者结果如下



### 1.2 创建一个矩阵

#### **Problem**

You need to create a matrix.



#### **Solution**

Use Numpy to create a two-dimensional array:

创建二维数组=矩阵

matrixExample.py

```python
# 导入库
import numpy as np

# 创建一个 matrix
matrix = np.array([[1, 2],
                   [1, 2],
                   [1, 2]])
```

![image-20220711204410624](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711204410624.png)

打印matrix如下



**两个原因不推荐适用matrix**

- First, arrays are the de facto standard data structure of NumPy.（数组是numpy标准的数据结构）
- Second the vast majority of NumPy operations return arrays, not matrix object.(大部分numpy的标准操作返回的是array)



### 1.3 Creating a Sparse Matrix（创建一个稀疏矩阵）

Sparse：稀疏

#### Problem
Given data with very few nonzero values, you want to efficiently represent it.

#### Solution
Create a sparse matrix:

sparseMatrixExample.py

```python

import numpy as np
# 引入sparse
from scipy import sparse

# 创建一个矩阵
matrix = np.array([[0, 0],
                  [0, 1],
                  [3, 0]])

# create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)

# view sparse matrix
print(matrix_sparse)
```

打印结果

![image-20220711210127903](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711210127903.png)

#### scipy库

- 需要额外安装，是一个开源的python高级科学计算库

  ```
  pip install scipy
  ```

  

- 额外支持的操作包括：数值积分、最优化、统计和一些专用函数

- 学习资源：[SciPy 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/scipy/scipy-tutorial.html)



#### Discussion

- A frequent situation in machine learning is having a huge amount of data; however most of the elements in the data are zeros. 机器学习中大多情形拥有大量数据但是数据很多时候为0



- Sparse matricies only store nonzero elements and assume all other values will be zero, leading to significant computational savings. 稀疏矩阵存储非零值并且假设其他值是0，从而节约计算量

- 稀疏矩阵的存储方式：CSR，存储非0坐标位置

  [Compressed Sparse Row（CSR）——稀疏矩阵的存储格式 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/342942385)

```python
# create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)

# view original sparse matrix
print(matrix_sparse)
```



打印结果：![image-20220711210219615](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711210219615.png)



- 稀疏矩阵中0元素的添加不会影响其在存储中所占的空间
- 稀疏矩阵有许多类型：compressed sparse column, list of lists, and dictionary of keys





### 1.4 **Selected Elements**

#### Problem

You need to select one or more elements in a vector or matrix.



#### Solution

NumPy's arrays make that easy

要求：访问特定的值

selectedExample.py

```python
import numpy as np

# create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# select the third element of vector
vector[2]
matrix[1,1]
```

结果：

![image-20220711211824539](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711211824539.png)

#### Discussion

- numpy的数组下标从0开始

-  With that caveat, NumPy offers a wide variety of methods for selecting (i.e., indexing and slicing) elements or groups of elements in arrays:（numpy数组提供了许多种访问方法）

  ```
  #访问全部元素
  vector[:]
  # 切片访问
  vector[:3]
  # 逆向访问
  vector[-1]
  # 访问前两行
  matrix[:2, :]
  # 访问所有行，第二列
  matrix[:,1:2]
  ```

结果：![image-20220711212718502](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711212718502.png)



### 1.5 Describing a Matrix

#### Problem

You want to describe the shape, size, and dimensions of the matrix



#### Solution

Use shape, size, and ndim:



描述矩阵信息

describeExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 行和列
print(matrix.shape)
#大小
print(matrix.size)
#维度
print(matrix.ndim)
```

结果：![image-20220711213423865](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711213423865.png)







### 1.6 Applying Operations to Elements

#### Problem

You want to apply some function to multiple elements in an array.



#### Solutions

Use NumPy's vectorize:



对多个元素进行函数操作

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix)

#创建一个函数
add_1000 = lambda i: i + 1000


# vectorized
vectorized_add_1000 = np.vectorize(add_1000)

# 适用该函数
vectorized_add_1000(matrix)

print(matrix)
```

运行结果

![image-20220711214052619](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711214052619.png)

#### Discusion

NumPy’s vectorize class converts a function into a function that can apply to all elements in an array or slice of an array. It’s worth noting that vectorize is essentially a for loop over the elements and does not increase performance. Furthermore, NumPy arrays allow us to perform operations between arrays even if their dimensions are not the same (a process called broadcasting). For example, we can create a much simpler version of our solution using broadcasting:

- vectorized可应用于数组和数组切片的所有元素

- 本质上vectorized是for循环不会提升性能

- 另外即使维度不同数组之间也可以进行操作，例如广播

  ```python
  matrix+1000
  ```

  

![image-20220711215615575](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711215615575.png)

### 1.7 Finding Maximum and Minimum Values

#### Problem

You need to find the maximum or minimum value in an array.



#### Solution

Use NumPy's max and min:

findingExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 最大值
np.max(matrix)
# 最小值
np.min(matrix)
```

结果：![image-20220711220443209](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711220443209.png)

#### Discussion

Often we want to know the maximum and minimum value in an array or subset of an array. This can be accomplished with the max and min methods. Using the axis parameter we can also apply the operation along a certain axis:

我们可以通过axis参数来求出每行或每列的最值

```python
# 每行最值
print(np.max(matrix, axis=0))
# 每列最值
print(np.max(matrix, axis=1))
```

结果![image-20220711221017316](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711221017316.png)



### 1.8 Calculating the Average, Variance, and Standard Deviation



#### Problem

You want to calculate some descriptive statistics about an array.



#### Solution

Use NumPy's mean, var, and std:



计算数组的统计数据

calculateExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# mean是算术平均值
np.mean(matrix)
# var是 方差
np.var(matrix)
# deviation 是标准差
np.std(matrix)
```

结果![image-20220711222210979](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711222210979.png)



#### Discussion

Just like with max and min, we can easily get descriptive statistics about the whole matrix or do calculations alon a single axis:

也可以像min和max一样可以指定axis：

```python
# find the mean value in each column
np.mean(matrix, axis=0)
```

结果：[4. 5. 6.]



### 1.9 Reshaping Arrays

#### Problem

You want to change the shape (number of rows and columns) of an array without changing the element values.



#### Solution

Use NumPy's reshape:

更改数组形状

reshapeExample.py

```python
# load library
import numpy as np

# create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# 重构
matrix.reshape(2, 6)
```

结果：![image-20220711222848134](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711222848134.png)

#### Discussion

- The only requirement is that the shape of the original and new matrix contain the same number of elements (i.e., the same size). We can see the size of a matrix using size。reshape要求重构前和重构后的size相等，拥有相同数量

- reshape可以用参数-1表示尽可能多

- Finally, if we provide one integer, reshape will return a 1D array of that length:（如果只有一个数字那么数组将变为1维）

  ```python
  print(matrix.size)
  print(matrix.reshape(1, -1))
  print(matrix.reshape(12))
  ```

  ![image-20220711223906150](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711223906150.png)

### 1.10 Transposing a Vector or Matrix

#### Problem

You need to transpose a vector or matrix



#### Solution

Use the T method:

转置

transposingExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 转置矩阵
print(matrix.T)
```

结果![image-20220711224157636](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711224157636.png)

- Transposing is a common operation in linear algebra where the column and row indices of each element are swapped. One nuanced point that is typically overlooked outside of a linear algebra class is that, technically, a vector cannot be transposed because it is just a collection of values:（普通向量无法转置）

- However, it is common to refer to transposing a vector as converting a row vector to a column vector (notice the second pair of brackets) or vice versa:（行向量可以转置）

  ```
  # 转置向量
  np.array([1, 2, 3, 4, 5, 6]).T
  # 转置 行向量
  np.array([[1, 2, 3, 4, 5, 6]]).T
  ```

  

结果![image-20220711224626255](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711224626255.png)



### 1.11 Flattening a Matrix

#### Problem

You need to transform a matrix into a one-dimensional array.



#### Solution

Use flatten:

平铺矩阵,使用flatten()函数

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# 平展矩阵
matrix.flatten()
```

输出：

![image-20220711225051358](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711225051358.png)

#### Discussion

flatten is a simple method to transform a matrix into a one-dimensional array. Alternatively, we can use reshape to create a row vector:

```python
#flatten等价于，但是不创建行向量
matrix.reshape(1, -1)
```



### 1.12 Finding the Rank of a Matrix

#### Problem

You need to know the rank of a matrix



#### Solution

Use NumPy's linear algebra method matrix_rank:

计算矩阵的秩

rankExample

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])

# 计算秩 
np.linalg.matrix_rank(matrix)
```

结果：2

#### Discussion

The rank of a matrix is the dimensions of the vector space spanned by its columns or rows. Finding the rank of a matrix is easy in NumPy thanks to matrix_rank.（求秩函数非常好用）



### 1.13 Calculating the Determinant

#### Problem

You need to know the determinant of a matrix



#### Solution

Use NumPy's linear algebra method det:

求行列式

determinantExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 计算行列式
np.linalg.det(matrix)
```

结果：0.0



#### Discussion
It can sometimes be useful to calculate the determinant of a matrix. NumPy makes this easy with det

计算行列式非常好用





### 1.14 Getting the Diagonal of a Matrix
#### Problem
You need to get the diagonal elements of matrix.

#### Solution
Use diagonal:

矩阵的对角线用diagonal函数

diagonalExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 对角线
print(matrix.diagonal())
```

结果：![image-20220711230149927](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711230149927.png)



#### Discussion

NumPy makes getting the diagonal elements of a matrix easy with diagonal. It is also possible to get a diagonal off from the main diagonal by using the offset parameter:

可以用offset函数获得副对角线

```python
print(matrix.diagonal(offset=1))
print(matrix.diagonal(offset=-1))
```

![image-20220711230424559](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711230424559.png)



### 1.15 Calculating the Trace of a Matrix

#### Problem

You need to calculate the trace of a matrix



#### Solution

Use trace:

计算矩阵的迹

traceExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 矩阵的迹
matrix.trace()
```

结果：1+4+9=14



#### Discussion

The trace of a matrix is the sum of the diagonal elements and is often used under the hood in machine learning methods. Given a NumPy multidimensional array, we can calculate the trace using trace. We can also return the diagonal of a matrix and calculate its sum:

- 矩阵的迹通常在底层适用
- 多维数组我们可以通过trace计算轨迹
- 等价于sum函数求对角线和

```python
#等价
print(sum(matrix.diagonal()))
```



### 1.16 Finding Eigenvalues and Eigenvectors

#### Problem

You need to find the eigenvalues and eigenvectors of a square matrix.



#### Solution

Use NumPy's linalg.eig:

特征值和特征向量

eigenExample.py

```
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
```

![image-20220711231812453](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711231812453.png)

#### Discussion

Eigenvectors are widely used in machine learning libraries. Intuitively, given a linear transformation represented by a matrix, $A$, eigenvectors are vectors that, when that transformation is applied, change only in scale (not direction). More formally:

$$A v = λ v$$

where $A$ is a square matrix, $λ$ contains the eigenvalues and $v$ contains the eigenvectors. In NumPy’s linear algebra toolset, ```eig``` lets us calculate the eigenvalues, and eigenvectors of any square matrix.

(解释特征值$λ$和特征向量$v$的定义)



### 1.17 Calculating Dot Products
#### Problem
You need to calculate the dot product of two vectors.

#### Solution
Use NumPy's dot:

向量点积

dotExample

```python
# load library
import numpy as np

# create two vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# 计算点积
print(np.dot(vector_a, vector_b))
print(vector_a@vector_b)
```

结果![image-20220711232328274](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711232328274.png)

#### Discussion

- The dot product of two vectors, a and b, is defined as:（点积定义）

  

  $$\sum(a_i * b_i)$$



- where $a_i$ is the ith element of vector a. We can use NumPy’s dot class to calculate the dot product. Alternatively, in Python 3.5+ we can use the new ```@``` operator:（3.5版本以上可以用@）

  ```python
  vector_a @ vector_b
  ```

  

### 1.18 Adding and Subtracting Matricies

#### Problem

You want to add or subtract two matricies



#### Solution

Use NumPy's add and subtract:

#### Discussion

Alternatively, we can simply use the + and - operators:

加法和减法

addAndSubstract.py

```python
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
```

![image-20220712102444640](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712102444640.png)

### 1.19 Multiplying Matricies

#### Problem

You want to multiply two matrices.

#### Solution

Use NumPy's dot:

#### Discussion

Alternatively, in Python 3.5+ we can use the @ operator:

矩阵点乘

```python
# load library
import numpy as np

# create matrices
matrix_a = np.array([[1, 1],
                     [1, 2]])

matrix_b = np.array([[1, 3],
                     [1, 2]])

# multiply two matrices
np.dot(matrix_a, matrix_b)
```

![image-20220711233348184](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711233348184.png)

### 1.20 Inverting a Matrix

#### Problem

You want to calculate the inverse of a square matrix.



#### Solution

Use NumPy's linear algebra inv method:

invertingExample.py

```python
# load library
import numpy as np

# create matrix
matrix = np.array([[1, 4],
                  [2, 5]])

# inv求逆
print(np.linalg.inv(matrix))

```

![image-20220711233623735](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711233623735.png)

#### Discussion

The inverse of a square matrix, $A$, is a second matrix $A^{–1}$, such that:



$A * A^{-1} = I$



where $I$ is the identity matrix. In NumPy we can use linalg.inv to calculate $A^{–1}$ if it exists. To see this in action, we can multiply a matrix by its inverse and the result is the identity matrix:

（定义：矩阵乘其逆矩阵得到单位矩阵）

```python
print(matrix @ np.linalg.inv(matrix))
```

![image-20220711233832891](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711233832891.png)



### 1.21 Generating Random Values

#### Problem

You want to generate pseudorandom values.



#### Solution

Use NumPy's random:

生成随机值

randomExample.py

```python
# load library
import numpy as np

# 设置种子
np.random.seed(0)

# 生成大小为3的随机数组
print(np.random.random(3))
```



#### Discussion

- NumPy offers a wide variety of means to generate random numbers, many more than can be covered here. In our solution we generated floats; however, it is also common to generate integers:

  (numpy提供了许多生成随机数的方法，可以生成整数)

- Alternatively, we can generate numbers by drawing them from a distribution:

  （我们可以从特殊分布中提取数字）

- Finally, it can sometimes be useful to return the same random numbers multiple times to get predictable, repeatable results. We can do this by setting the “seed” (an integer) of the pseudorandom generator. Random processes with the same seed will always produce the same output. We will use seeds throughout this book so that the code you see in the book and the code you run on your computer produces the same results.

（有时可以通过设置相同的种子来产生多次相同的值，这有时候会很有用；种子产生的是伪随机数）

```python
# 生成3个在 1 和 10的随机整数
print(np.random.randint(0, 11, 3))
# 从均值为0的正态分布生成三个随机数
# 方差为1
print(np.random.normal(0.0, 1.0, 3))
# 从logistic分布中获得3个随机数
print(np.random.logistic(0.0, 1.0, 3))
# 从均值分布中获得3个随机数
print(np.random.uniform(1.0, 2.0, 3))
```

![image-20220711235305009](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220711235305009.png)







## Chapter 2 Loading Data

#### 2.0 Introduction

The first step in any machine learning endeavor is to get the raw data into our system. The raw data might be a logfile, dataset file, or database. Furthermore, often we will want to retrieve data from multiple sources. The recipies in this chapter look at methods of loading data from a variety of sources, including CSV files and SQL databases. We also cover methods of generating simulated data with desirable properties for experimentation. Finally, while there are many ways to load data in the Python ecosystem, we will focus on using the pandas library's extensive set of methods for loading external data, and using scikit-learn--an open source machine learning library in Python--for generating simulated data.

总结：

- 任何机器学习努力的第一步都是将原始数据输入我们的系统。
- 我们希望从多个数据源获得数据（pandas）
- 我们还可以通过工具生成数据（scikit-learn ）

#### 2.1 Loading a Sample Dataset

#### Problem

You want to load a prexisting sample dataset

#### Solution

scikit-learn comes with a number of popular datasets for you to use:

加载一个先前已经存在的数据源
sampleExample.py

```python
# load scikit-learn's datasets
from sklearn import datasets

# 加载 digits 数据集
digits = datasets.load_digits()

# 创建 features matrix
features = digits.data
print(features)
# 创建 target vector
target = digits.target
print(target)
#  查看第一个 observation
print(features[0])
```

![image-20220712111840421](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712111840421.png)

样例代码中含有scikit-learn库，需要单独安装

- 关于scikit-learn

```
#使用anaconda安装 4.12.0版本
conda install scikit-learn
```

scikit-learn，又写作sklearn，是一个开源的基于python语言的机器学习工具包。它通过NumPy, SciPy和Matplotlib等python数值计算的库实现高效的算法应用，并且涵盖了几乎所有主流机器学习算法。

scikit-learn中文社区：[scikit-learn中文社区](https://scikit-learn.org.cn/)

- 关于datasets中的数据集

  ```
  datasets.load_boston #波士顿房价数据集  
  datasets.load_breast_cancer #乳腺癌数据集  
  datasets.load_diabetes #糖尿病数据集  
  datasets.load_digits #手写体数字数据集  
  datasets.load_files  
  datasets.load_iris #鸢尾花数据集  
  datasets.load_lfw_pairs  
  datasets.load_lfw_people  
  datasets.load_linnerud #体能训练数据集  
  datasets.load_mlcomp  
  datasets.load_sample_image  
  datasets.load_sample_images  
  datasets.load_svmlight_file  
  datasets.load_svmlight_files  
  ```

  本例子使用的是手写体数字数据集

- 关于features matrix和target vector

  features matrix：特征数据数组

  target vector：标签数组

- 关于术语observation

  Observation 

  A single unit in our level of observation—for example, a person, a sale, or a record.

​	observation理解下来应该是观测值的意思



#### Discussion

Often we do not want to go through the work of loading, transforming and cleaning a real-world dataset before we can explore some machine learning algorithm or method. Luckily, scikit-learn comes with some common datasets we can quickly load. These datasets are often called "toy" datasets because they are far smaller and cleaner than a dataset we would see in the real world. Some popular sample datasets in scikit-learn are:（给出了一些小型数据集）

```
load_boston
```

* Contains 503 observations on Boston housing prices. It is a good dataset for exploring regression algorithms.（包含503个观测值的波士顿房价数据集）

```
load_iris
```

* Contains 150 observations on the measurements of Iris flowers. It is a good dataset for exploring classification algorithms（150个样例的鸢尾花数据集 ）

```
load_digits
```

* Cotnains 1,797 observations from images of handwritten digits. It is a good dataset for teaching image classification（手写数字数据集）
* 其他的数据集见上方



### 2.2 Creating a Simulated Dataset

#### Problem

You need to generate a dataset of simulated data

#### Solution

scikit-learn offers any methods for creating simulated data. Of those, three methods are particularly useful

When we want a dataset designed to be used with linear regression, `make_regression` is a good choice:

要求：生成模拟的数据集

##### 线性回归数据集函数：`make_regression`

make_regressionExample.py

```python
# load library
from sklearn.datasets import make_regression

# 生成 features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples=100,  # 样本数量
                                                 n_features=3,  # 特征
                                                 n_informative=3,  # 参与建模的特征数
                                                 n_targets=1,   # 因变量个数
                                                 noise=0.0,     # 噪声
                                                 coef=True,     # 是否输出coef标志
                                                 random_state=1)    # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature Matrix \n {}".format(features[:3]))
print("Target Vector \n {}".format(target[:3]))

```

![image-20220712111900975](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712111900975.png)

##### 分类数据集：`make_classification`:

make_classificationExample.py

```python
# load library
from sklearn.datasets import make_classification

# generate features matrix and target vector

features, target = make_classification(n_samples = 100,  # 样本个数
                                       n_features = 3,      # 特征数
                                       n_informative = 3,   # 参与建模的特征数
                                       n_redundant = 0,     # 冗余信息
                                       n_classes = 2,       # 类的个数
                                       weights = [.25, .75],    # 权重
                                       random_state = 1)        # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature matrix\n {}".format(features[:3]))
print("Target vector\n {}".format(target[:3]))
```

![image-20220712112429938](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712112429938.png)

##### 聚类数据集`make_blobs`

make_blobsExample.py

```python
# load library
from sklearn.datasets import make_blobs

# generate feature_matrix and target vector
features, target = make_blobs(n_samples=100,  # 样本数量
                              n_features=2,  # 特征数量
                              centers=3,  # 类别数（中心数）
                              cluster_std=0.5,  # 每个类的方差
                              shuffle=True,  # 是否洗乱数据
                              random_state=1)  # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature Matrix\n {}".format(features[:3]))
print("Target Vector\n {}".format(target[:3]))

```

![image-20220712112920611](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712112920611.png)

#### Discussion

- As might be apparent from the solutions, make regression returns a feature matrix of flaot values and a target vector of float values, while make_classification and make_blobs return a feature matrix of float values and a target vector of integers representing membership in a class.

  (make_regression返回浮点值的特征矩阵和浮点值的目标向量，而 make_classification 和 make_blobs 返回浮点值的特征矩阵和表示类成员资格的整数目标向量。 )

- scikit-learn's simulated datasets offer extensive options to control the type of data generated.

  (scikit-learn提供广泛选择来构建数据集)

- In `make_regression` and `make_classification`, `n_informative` determines the number of features that are used to generate the target vector. If n`_informative` is less than the totla number of features (`n_features`), the resulting dataset will have redundant features that cna be identified through feature selection techniques

  （在 `make_regression` 和 `make_classification` 中，`n_informative` 决定了用于生成目标向量的特征数量。如果 n`_informative` 小于特征总数 (`n_features`)，则生成的数据集将具有冗余特征，这些特征可以通过特征选择技术识别）

- In addition, `make_classification` contains a `weights` parameter that allows us to simulate datasets with imbalanced classes. For example, `weights = [.25, .75]` would return a dataset with 25% of observations belonging to one class and 75% to the other

  （`make_classification` 包含一个`weights` 参数，允许我们模拟具有不平衡类的数据集。例如，`weights = [.25, .75]` 将返回一个数据集，其中 25% 的观察属于一个类，75% 属于另一个） 

- For `make_blobs`, the centers parameter determines the number of clusters generated. Using the `matplotlib` visualization library we can visualize the clusters generated by `make_blobs`:

对于“make_blob”，centers 参数决定了生成的簇数。使用 `matplotlib` 可视化库，我们可以可视化 `make_blobs` 生成的集群：

需要安装matplotlib库

```
conda install matplotlib
```



```python
# load library
from sklearn.datasets import make_blobs

# load library
import matplotlib.pyplot as plt

# generate feature_matrix and target vector
features, target = make_blobs(n_samples=100,  # 样本数量
                              n_features=2,  # 特征数量
                              centers=3,  # 类别数（中心数）
                              cluster_std=0.5,  # 每个类的方差
                              shuffle=True,  # 是否洗乱数据
                              random_state=1)  # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature Matrix\n {}".format(features[:3]))
print("Target Vector\n {}".format(target[:3]))


# view scatterplot
plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()
```

![image-20220712114435508](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712114435508.png)



### 2.3 Loading a CSV File

#### Problem

You need to import a comma-separated values (CSV) file.



#### Solution

Use the `pandas` library's `read_csv` to load a local or hosted CSV file:

需要安装pandas

```
conda install pandas
```

[Pandas 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/pandas/pandas-tutorial.html)

loadCSVExample.py

```python
# load library
import pandas as pd

# create url


# 加载数据
df = pd.read_csv("data.csv")

print(df.head(2))
```

因为无法打开课本中的csv文件

所以使用一个本地csv文件

得到结果

![image-20220712145328454](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712145328454.png)

data.csv:![image-20220712152509510](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712152509510.png)

#### Discussion

- 加载之前快速查看文件内容通常很有用
- read_csv 有 30 多个参数，因此文档可能令人生畏。这些参数主要是为了让它能够处理各种 CSV 格式。
  - pandas 的 sep 参数允许我们定义文件中使用的分隔符。
  - header 参数允许我们指定标题行是否存在或存在于何处。如果标题行不存在，我们设置 header=None。



### 2.4 Loading an Excel File

#### Problem

You need to import an Excel spreadsheet



#### Solution

Use the `pandas` library's `read_excel` to load an Excel spreadsheet:

用pandas打开excel文件

loadExcelExample.py

```python
import pandas as pd

import ssl
# Python 从 2.7.9版本开始，就默认开启了服务器证书验证功能，如果证书校验不通过，则拒绝后续操作；这样可以防止中间人攻击，并使客户端确保服务器确实是它声称的身份。如果是自签名证书，由于一般系统的CA证书中不存在在自签名的CA证书内容，从而导致证书验证不通过。
ssl._create_default_https_context = ssl._create_unverified_context


# 因为原书的excel无法访问，所以替换了一个url
url = "https://www.sample-videos.com/xls/Sample-Spreadsheet-10-rows.xls"

# 加载url
df = pd.read_excel(url, sheet_name=0, header=None)

# 打印前两行
print(df.head(2))
```

结果：![image-20220712154958428](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712154958428.png)



#### Discussion

- 附加参数 sheetname，它指定我们希望加载 Excel 文件中的哪个工作表。
- 如果我们需要加载多张工作表，请将它们作为列表包含在内。 例如， sheetname= [0,1,2, "Monthly Sales"] 将返回包含第一张、第二张和第三张工作表以及名为 Monthly Sales 的工作表的 pandas DataFrame 字典。



### 2.5 Loading a JSON File

#### Problem

You need to load a JSON file for data preprocessing



#### Solution

The pandas library provides `read_json` to convert a JSON file a pandas object:

加载json文件，使用```read_json```

```python
# load library
import pandas as pd

# create url
url = 'https://raw.githubusercontent.com/domoritz/maps/master/data/iris.json'

# load data
df = pd.read_json(url, orient="columns")

# view first two rows
print(df.head(2))
```

![image-20220712160321314](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712160321314.png)

#### Discussion

- orient 参数，它向 pandas 指示 JSON 文件的结构
- pandas 提供的另一个有用的工具是 json_normalize，它可以帮助将半结构化 JSON 数据转换为 pandas DataFrame。



###  2.6 Querying a SQL Database

#### Problem

You need to load data from a databaseu sing structured query language (SQL)



#### Solution

`pandas`' `read_sql_query` allows us to make a SQL query to a database and load it:

读取sql中的内容

loadSqlExample.py

```python
import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接
# 按实际情况依次填写MySQL的用户名、密码、IP地址、端口、数据库名
engine = create_engine('mysql+pymysql://root:444555@localhost:3306/lab5')

sql_query = 'select * from student;'
# 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
df_read = pd.read_sql_query(sql_query, engine)
print(df_read)
```

![image-20220712163229421](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712163229421.png)

（原书使用sqlite，本例子改成了mysql）

#### Discussion

- ```create_engine```创建一个mysql的数据连接
- ```read_sql_query```将结果放到DataFrame







## Chapter 3 Data Wrangling（数据整理）



### 3.0 Introduction

- Data wrangling is a broad term used, often informally, to describe the process of transforming raw data to a clean and organized format ready for use.

  （数据整理（data wrangling)指将数据转换为可供使用的干净且有组织的格式组织）

- The most common data structure used to "wrangle" data is the data frame, which can be both intuitive and incredibly versatile. Data frames are tabular, meaning that htey are based on rows and columns like you'd find in a spreadsheet

​	（用于“整理”数据的最常见数据结构是data frame，它既直观又非常通用。



对于书中的例子：

```python
# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/titanic-csv'
# Load data as a dataframe
dataframe = pd.read_csv(url)
# Show first 5 rows
dataframe.head(5)

```

![image-20220712165426371](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220712165426371.png)

需要注意3点：

- 首先，在data frame中，每一行对应一个观察值（例如，一名乘客），每一列对应一个特征（性别、年龄等）。例如，通过查看第一个observation，我们可以看到 Elisabeth Walton Allen 小姐留在头等舱，29 岁，是女性，并且在灾难中幸存下来。 
- 其次，每列包含一个名称（例如，姓名、PClass、年龄），每行包含一个索引号（例如，幸运的伊丽莎白沃尔顿艾伦小姐为 0）。 我们将使用这些来选择和操作观察和特征。
-  第三，Sex 和 SexCode 两列包含不同格式的相同信息。在 Sex 中，女性用字符串 female 表示，而在 SexCode 中，女性用整数 1 表示。我们希望所有特征都是唯一的，因此我们需要删除其中一列。 在本章中，我们将介绍使用 pandas 库操作数据帧的各种技术，目的是创建一个干净、结构良好的观察集以供进一步预处理。

### 3.1 Creating a Data Frame

#### Problem 

You want to create a new data frame.

#### Solution

pandas has many methods of creating a new DataFrame object. One easy method is to create an empty data frame using DataFrame and then define each column separately:

pandas拥有许多methods来创建新的DataFrame 

emptyDataFrameExample.py

```python

import pandas as pd
# Create DataFrame
dataframe = pd.DataFrame()
# 用字典的方式添加新的一行
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]
# 展示dataframe
print(dataframe)

# 创建新的一行
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name', 'Age', 'Driver'])
# 拼接一行
dataframe = dataframe.append(new_person,ignore_index=True)
print(dataframe)
```

![image-20220714095146021](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714095146021.png)

**值得注意的是未来append属性要被废除，所以最好还是创建小的dataframe然后用concat拼接到总的Dataframe中**



#### Discussion

- pandas库提供无数种创建DataFrame的方法
- 现实中常常采用从其他来源产生一个DataFrame而不是创建一个新的DataFrame然后填充





### 3.2 Describing the Data

查看DataFrame的相关信息

DescribeExample.py

```python
import pandas as pd
# 因为无法访问国外的csv文件，使用国内的网站代替
url = 'https://www.gairuo.com/file/data/dataset/GDP-China.csv'
df = pd.read_csv(url)
# show first two rows
print(df.head(2))  # also try tail(2) for last two rows

# show dimensions
print("Dimensions: {}".format(df.shape))

# show statistics
print(df.describe())

```



![image-20220714101813747](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714101813747.png)



#### Discussion

- 由于数据量过大，为了能够更好的访问数据，需要了解数据类型和结构，这就需要获取小的切片和获取统计信息
- 一些数字列往往代表类别或者其他枚举类信息，这样的信息的统计信息往往没有意义，例如性别由0和1表示，而他的方差往往没有统计意义



### 3.3 Navigating DataFrames

需要选取单个数据或者数据切片

navigateExample.py

```python
# Load library
import pandas as pd
# Create URL
url = 'titanic.csv'
# Load data
dataframe = pd.read_csv(url)
# Select first row
print(dataframe.iloc[0])
print()
# Select three rows
print(dataframe.iloc[1:4])


# Set index
dataframe = dataframe.set_index(dataframe['Lname'])
print(dataframe.loc['Braund'])
```

![image-20220714105729148](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714105729148.png)

#### Discussion

- pandas创建的dataframe都含有索引，默认是一个整数
- DataFrame可以设置唯一的字母数字字符串作为索引
- loc可以根据自定义的标签来返回对应的元素
- iloc通过位置来返回对应的一行
- loc和iloc是非常有用的数据清理函数



### 3.4 Selecting Rows Based on Conditionals

查找某些行元素

selectExample.py

```python
# 引入库
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 单条件查询

print(dataframe[dataframe['Sex'] == 'female'].head(2))

# 多条件查询
print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 50)])
```

![image-20220714111040285](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714111040285.png)



#### Discussion

- 有效使用条件筛选和过滤是数据清理的重要任务之一



### 3.5 Replacing Values

目标：替换指定列的值

replaceExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 替换female 为male
print(dataframe['Sex'].replace("female", "Woman").head(2))

# 替换 "female" and "male 为 "Woman" and "Man"
print(dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))
```

![image-20220714111935628](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714111935628.png)

### 3.6 Renaming Columns

重命名属性

renameExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 替换一列
print(dataframe.rename(columns={'Pclass': 'Passenger Class'}).head(2))


# 同时替换两列
print(dataframe.rename(columns={'Pclass': 'Passenger Class','Lname': 'Last Name'}).head(2))

```

![image-20220714112643564](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714112643564.png)

#### Discussion

- 通过字典来重命名是首选方法

- 可以通过集合来一次性设置所有列，例：

  ```python
  # Load library
  import collections
  # Create dictionary
  column_names = collections.defaultdict(str)
  # Create keys
  for name in dataframe.columns:
  	column_names[name]
  # Show dictionary
  column_names
  defaultdict(str,
  {'Age': '',
  'Name': '',
  'PClass': '',
  'Sex': '',
  'SexCode': '',
  'Survived': ''})
  
  ```

  

### 3.7 Finding the Min, Max, Sum, Average, and Count

查找最大最小、和、平均值和出现次数

statisticExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)
# 计算统计属性值
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
```

![image-20220714113327699](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714113327699.png)

#### Discussion

- 除了解决方案中使用的统计数据，pandas 还提供方差（var）、标准差（std）、峰度（kurt）、偏度（skew）、均值的标准误差（sem）、众数（mode）、中位数（median )，以及其他一些。

- 可以直接作用于整个DataFrame

  ```python
  import pandas as pd
  
  url = 'titanic.csv'
  
  dataframe = pd.read_csv(url)
  # 计算全部属性的次数
  
  print(dataframe.count())
  ```

  ![image-20220714113359209](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714113359209.png)

### 3.8 Finding Unique Values

查重

uniqueExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 查看所有可能的值，返回一个数组
print(dataframe['Sex'].unique())

# 显示次数
print(dataframe['Sex'].value_counts())
```

![image-20220714115438059](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714115438059.png)

#### Discussion

- unique 和 value_counts 对于操作和探索分类列都很有用。 很多时候，在分类列中会有需要在数据整理阶段处理的类。
- value_counts会出现问题：当出现某种不合规的“类”时，往往这些类的统计数据是不需要的：(例如该图中的*)

![image-20220714120051309](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714120051309.png)

- 可以使用nunique()来查看有多少种不一样的类别



### 3.9 Handling Missing Values

处理null的值

nullExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

## Select missing values, show two rows
print(dataframe[dataframe['Age'].isnull()].head(2))

```

![image-20220714141457657](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714141457657.png)

#### Discussion

- 缺失值是数据整理中普遍存在的问题，但许多人低估了处理缺失数据的难度。 pandas 使用 NumPy 的 NaN（“非数字”）值来表示缺失值，但重要的是要注意 NaN 在 pandas 中并没有完全实现。

例如：

```python
# Attempt to replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
```

结果：

NameError Traceback (most recent call last) in () 1 # Attempt to replace values with NaN ----> 2 dataframe['Sex'] = dataframe['Sex'].replace('male', NaN) NameError: name 'NaN' is not defined

- 为了拥有 NaN 的全部功能，我们需要首先导入 NumPy 库：

  ```python
  # Load library
  import numpy as np
  # Replace values with NaN
  dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)
  ```

  

- 通常，数据集使用特定值来表示缺失的观察值，例如 NONE、-999 或 .. pandas 的 read_csv 包含一个参数，允许我们指定用于表示缺失值的值：

  ```
  # Load data, set missing values
  dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])
  ```

  

### 3.10 Deleting a Column

删除1列,

调用函数drop()

deleteColExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 删除age
print(dataframe.drop('Age', axis=1).head(2))

# 删除两列
print(dataframe.drop(['Age', 'Sex'], axis=1).head(2))

# 通过列的description删除
print(dataframe.drop(dataframe.columns[1], axis=1).head(2))

```

![image-20220714143225282](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714143225282.png)

#### Discussion

- 不推荐使用del dataframe['Age']方法（因为他的底层实现方式）

  **查阅资料：**[(87条消息) #深入分析# pandas中使用 drop 和 del删除列数据的区别_energy_百分百的博客-CSDN博客_tensorflow删除列](https://blog.csdn.net/lch551218/article/details/113844450)

  1、del是内置函数

  2、drop可以同时操作多个项目效率高

  3、drop更加灵活。可以在本地操作也可以返回副本

- 不推荐调用pandas库函数的时候使用inplace=True的参数，这可能会导致更复杂的数据处理管道出现问题，因为我们将 DataFrame 视为可变对象（它们在技术上是可变对象）。应该将DataFrame使用时视为不可变对象

- ``` #Create a new DataFrame``` 

  ``` dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)```

就是一个例子，如果将DataFrames视为不可变对象，那将减少很多麻烦



### 3.11 Deleting a Row

删除一行

deleteRowExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)
# 删除female的前两列
print(dataframe[dataframe['Sex'] != 'male'].head(2))
```

![image-20220714144619752](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714144619752.png)

#### Discussion

- 可以使用drop函数来实现删除行，但是更实用的方法将条件包装在dataframe[]中
- 可以通过索引来删除



### 3.12 Dropping Duplicate Rows

删除重复的行

现在样例中添加重复的行

![image-20220714145503033](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714145503033.png)

dropDupRowsExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 去除重复的行
print(dataframe.drop_duplicates().head(2))

#检查行数
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

```

![image-20220714145547984](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714145547984.png)

![image-20220714145832547](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714145832547.png)

重复的行被删除了



#### Discussion

- 该解决方案并没有删除任何行。原因是因为 drop_duplicates 默认只删除在所有列中完全匹配的行。

- 通常我们想要筛选数据可以通过子集来检查行

  ```python
  dataframe.drop_duplicates(subset=['Sex'])
  ```

  

- 可以通过keep参数来保留重复行的第一次出现

  ```python
  # Drop duplicates
  dataframe.drop_duplicates(subset=['Sex'], keep='last')
  ```

- 还有一个相关的函数是```duplicate```可以判断这行是不是重复的，可以完成一些复杂的筛选工作





### 3.13 Grouping Rows by Values

组操作

groupExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean
# of each group
print(dataframe.groupby('Sex').mean())

# 对某列计数
print(dataframe.groupby('Survived')['Name'].count())
# 对某列求平均值
print(dataframe.groupby(['Sex','Survived'])['Age'].mean())
```

![image-20220714151257555](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714151257555.png)

#### Discussion

- groupby是数据清理真正的起始点
- groupby往往需要搭配统计类函数
- 可以通过字典的方式对单一列进行统计



### 3.14 Grouping Rows by Time

按照日期进行分组

timeGroupExample.py

```python
# Load libraries
import pandas as pd
import numpy as np
# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
# Create DataFrame
dataframe = pd.DataFrame(index=time_index)
# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)
# Group rows by week, calculate sum per week
print(dataframe.resample('W').sum())
```

![image-20220714151608578](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714151608578.png)

#### Discussion

- resample要求是数据集的索引是一个类似时间属性的值
- resample常用的一个参数可以指定时间间隔'W'表示周，‘M'表示月,还可以配置一定的比例例如’2W'
- resample默认返回时间组的“右边缘”也就是最小值,例如上述例子2017-06-11为右边缘
- 详细的讲解可以看[python时序分析之重采集（resample） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/70353374)



### 3.15 Looping Over a Column

迭代某一列的所有元素

LoopingColExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())
# Show first two names uppercased
print([name.upper() for name in dataframe['Name'][0:2]])

```



![image-20220714153106755](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714153106755.png)

#### Discussion

- 可以用列表的方式进行访问
- 下一节的apply方法更加常用



### 3.16 Applying a Function Over All Elements in a Column

对某一列元素使用函数

applyExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 大写函数
def uppercase(x):
    return x.upper()
# apply作用后的前两行
print(dataframe['Name'].apply(uppercase)[0:2])
```

![image-20220714153654683](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714153654683.png)

#### Discussion

- 作者评价apply是个好函数

### 3.17 Applying a Function to Groups

对分组后的元素进行apply操作

applyGroupExample.py

```python
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Group rows, apply function to groups
print(dataframe.groupby('Sex').apply(lambda x: x.count()))
```

![image-20220714153907315](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714153907315.png)

#### Discussion

- 作者评价apply和group一起很有用



### 3.18 Concatenating DataFrames

连接两个数据帧

使用pandas.concat函数

concatExample.py

```python
# Load library
import pandas as pd
# Create DataFrame
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])
# Create DataFrame
data_b = {'id': ['4', '5', '6'],
'first': ['Billy', 'Brian', 'Bran'],
'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])
# 连接行
print(pd.concat([dataframe_a, dataframe_b], axis=0))
```

![image-20220714154450269](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714154450269.png)

此外还可以连接列

```python
print(pd.concat([dataframe_a, dataframe_b], axis=1))
```

![image-20220714154536863](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714154536863.png)

#### Discussion

- concatenating——将两个对象粘合在一起，通过axis来指示方向
- 可以使用append凭借series(前面资料查阅发现新版本的pandas会废弃append函数)



### 3.19 Merging DataFrames

合并两个DataFrrames

mergeExample.py

```
# Load library
import pandas as pd

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                          'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns=['employee_id',
                                                           'name'])
# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id', 'total_sales'])
# 自然连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id'))

# 外连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer'))

# 左连接
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'))

# 指定属性链接
print(pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id'))

```

![image-20220714155158318](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220714155158318.png)

#### Discussion

- merge与数据库中的join非常相似
- 支持的几种连接方式
  - inner： 仅返回在两个 DataFrame 中匹配的行（例如，返回任何行 在 dataframe_employees 和 dataframe_sales 中都有一个employee_id 值）。
  - outer： 返回两个 DataFrame 中的所有行。如果一行存在于一个 DataFrame 中但不存在于另一个 DataFrame 中，则为缺失值填充 NaN 值（例如，返回 employee_id 和 dataframe_sales 中的所有行）。
  - left： 返回左侧 DataFrame 中的所有行，但仅返回右侧的行 与左侧 DataFrame 匹配的 DataFrame。为缺失值填充 NaN 值（例如，返回 dataframe_employees 中的所有行，但仅返回 dataframe_sales 中具有出现在 dataframe_employees 中的employee_id 值的行）。 ‘
  - right： 返回右侧 DataFrame 中的所有行，但仅返回左侧的行 与正确 DataFrame 匹配的 DataFrame。为缺失值填充 NaN 值（例如，返回 dataframe_sales 中的所有行，但仅返回 dataframe_employees 中具有出现在 dataframe_sales 中的employee_id 值的行）。

​          这些方式通过outer指定

- 另外还可以利用`left_on`和`right_on`完成一些混合连接

## Chapter 4 Handling Numerical Data



### 4.0 Introduction

Quantitative data is the measurment of something--weather class size, monthly sales, or student scores. The natural way to represent these quantities is numerically (e.g., 20 students, $529,392 in sales). In this chapter we will cover numerous strategies for transforming raw numerical data into features purpose-built for machine learning algoristshms

- 在本章中，我们将介绍许多将原始数值数据转换为专为机器学习算法构建的特征的策略

### 4.1 Rescaling a feature

Use scikit-learn's `MinMaxScaler` to rescale a feature array

特征缩放是什么？特征缩放的目标就是数据规范化，使得特征的范围具有可比性。它是数据处理的[预处理](https://so.csdn.net/so/search?q=预处理&spm=1001.2101.3001.7020)处理，对后面的使用数据具有关键作用。



rescalingExample.py

```python
# Load libraries
import numpy as np
from sklearn import preprocessing

# 创建特征矩阵
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])
# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)
# Show feature
print(scaled_feature)

```

![image-20220715152220045](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715152220045.png)

#### Discussion

#### Discussion

Rescaling is a common preprocessing task in machine learning. Many of the algorithms described later in this book will assume all features are on the same scale, typically 0 to 1 or -1 to 1. There are a number of rescaling techniques, but one of the simlest is called **min-max scaling**. Min-max scaling uses the minimum and maximum values of a feature to rescale values to within a range. Specfically, min-max calculates:

- 特征缩放是机器学习中常见的预处理任务。
- 本书后面描述的许多算法将假设所有特征都处于相同的比例，通常是 0 到 1 或 -1 到 1。有许多重新缩放技术
- 本节使用的是最简单的一种称为**min-max scaling的技术**

$$x_i‘  = \frac{x_i - min(x)}{max(x) - min(x)}$$



where x is the feature vector, $x_i$ is an individual element of feature x, and $x_i^`$ is the rescaled element





### 4.2 Standardizing a Feature

scikit-learn's `StandardScaler` transforms a feature to have a mean of 0 and a standard deviation of 1.

standardScalerExample.py

```python
import numpy as np
from sklearn import preprocessing

feature = np.array([
    [-1000.1],
    [-200.2],
    [500.5],
    [600.6],
    [9000.9]
])

scaler = preprocessing.StandardScaler()

# 标准化
standardized = scaler.fit_transform(feature)

print(standardized)
```

#### ![image-20220715152805484](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220715152805484.png)



#### Discussion

- 标准化使的feature的均值 $\bar x$为0，$\sigma$为1
- $$x_i^` = \frac{x_i - \bar x}{\sigma}$$
- 标准化是机器学习预处理的常用缩放方法，比最大最小化法用的多。但是还是建议在神经网络中使用最大最小化法，不使用标准化
- 如果我们的数据有明显的异常值，它会通过影响特征的均值和方差来对我们的标准化产生负面影响。在这种情况下，使用中位数和四分位数范围重新调整特征通常会有所帮助。在 scikit-learn 中，我们使用 *RobustScaler* 方法执行此操作：

```python
import numpy as np
from sklearn import preprocessing

feature = np.array([
    [-1000.1],
    [-200.2],
    [500.5],
    [600.6],
    [9000.9]
])

# scaler = preprocessing.StandardScaler()
#
# # 标准化
# standardized = scaler.fit_transform(feature)
#
# print(standardized)

# create scaler
robust_scaler = preprocessing.RobustScaler()

# 中值代替
robust = robust_scaler.fit_transform(feature)

print(robust)
```

![image-20220715155733760](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715155733760.png)

### 4.3 Normalizing Observations

Use scikit-learn's `Normalizer` to rescale the feature values to have unit norm (a total length of 1)

observationNormalizeExample.py

```python
import numpy as np
from sklearn.preprocessing import Normalizer

# create feature matrix
features = np.array([
    [0.5, 0.5],
    [1.1, 3.4],
    [1.5, 20.2],
    [1.63, 34.4],
    [10.9, 3.3]
])

# create normalizer
normalizer = Normalizer(norm="l2")

# normalize matrix
print(normalizer.transform(features))

```

![image-20220715160050862](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220715160050862.png)

#### Discussion

- `Normalizer` 根据参数将单个观测值的值重新调整为具有单位的范式（它们的长度之和为 1）。

- `Normalizer` 提供了三个范式选项，欧几里得范数（通常称为 L2）是默认值： $$ ||x||_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2} $$
- 曼哈顿范数 (L1)： $$ ||x||_1 = \sum_{i=1}^n{x_i} $$
- 实际上，请注意 `norm='l1'` 会重新调整观察值，使其总和为 1，这有时可能是理想的质量

```python
# transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)
print("Sum of the first observation's values: {}".format(features_l1_norm[0,0] + features_l1_norm[0,1]))
```

结果：Sum of the first observation's values: 1.0

- **查寻资料得到第三种参数为max,若为max时，样本各个特征值除以样本中特征值最大的值**



### 4.4 Generating Polynomial and Interaction Features

#### Problem

创建多项式特征

#### Solution

使用scikit-learn里的`built-in`函数

polynomialExample.py

```python
# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])
# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
# 创建多项式特征
print(polynomial_interaction.fit_transform(features))

# 只出现交叉项
interaction = PolynomialFeatures(degree=2,
                                 interaction_only=True, include_bias=False)
print(interaction.fit_transform(features))
```

![image-20220715164240925](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715164240925.png)

#### Discussion

- 什么是多项式特征？为什么要创建多项式特征

  多项式特征可以理解成现有特征的乘积。当我们想要包含特征与目标之间存在非线性关系的概念时，通常会创建多项式特征。

- 此外，我们经常会遇到一个特性的效果依赖于另一个特性的情况。每个特征对目标（甜度）的影响是相互依赖的。我们可以通过包含一个交互特征来对这种关系进行编码，该交互特征是各个特征的产物。



### 4.5 Transforming Features 

对一组特征进行自定义的转换

在 scikit-learn 中，使用 FunctionTransformer 将函数应用于一组 特征：

transformingExample.py

```python
# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Load library
import pandas as pd

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])


# 创建一个函数
def add_ten(x):
    return x + 10


# Create transformer
ten_transformer = FunctionTransformer(add_ten)
# 运用transformer
print(ten_transformer.transform(features))


# 将features创建为DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# 使用第三章的apply函数
df.apply(add_ten)

```

![image-20220715170752238](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220715170752238.png)



#### Discussion

- 通常希望对一项或多项功能进行一些自定义转换。
- add_ten是一个很简单的函数，但是实际应用种我们可以使用transformer或者apply进行更复杂的函数的应用



### 4.6 Detecting Outliers

检测异常值

detectOutliersExample.py

```python
# Load libraries
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# 创建模拟值
features, _ = make_blobs(n_samples=10,
                         n_features=2,
                         centers=1,
                         random_state=1)
# 将第一行的值替换为一个极端的值
features[0, 0] = 10000
features[0, 1] = 10000
# 创建一个detector
outlier_detector = EllipticEnvelope(contamination=.1)
# 拟合 detector
outlier_detector.fit(features)
# 预测 outliers
print(outlier_detector.predict(features))


# 创建 单一 feature
feature = features[:, 0]

# 创建函数计算iqr并且返回iqr外的值
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))


# Run function
print(indicies_of_outliers(feature))
```



结果：

![image-20220715205647146](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715205647146.png)

本案例主要展现两种检测异常值的方法

`EllipticEnvelope`	

- 假设全部数据可以表示成基本的多元高斯分布（正态分布），EllipticEnvelope函数试图找出数据总体分布关键参数。尽可能简化算法背后的复杂估计，可认为该算法主要是**检查每个观测量与总均值的距离**。
- Covariance.EllipticEnvelope函数使用时需要考虑污染参数(contamination parameter) ，该参数是**异常值在数据集中的比例**，默认取值为0.1，最高取值为0.5。
- 缺陷：
  - EllipticEnvelope函数适用于有控制参数的高斯分布假设，使用时要注意：非标化的数据、二值或分类数据与连续数据混合使用可能引发错误和估计不准确。
  - EllipticEnvelope函数假设全部数据可以表示成基本的多元高斯分布，当数据中有多个分布时，算法试图将数据适应一个总体分布，倾向于寻找最偏远聚类中的潜在异常值，而忽略了数据中其他可能受异常值影响的区域。

`IQR`:

- 四分位距法
- QL：下四分位数，表示全部观察值中有四分之一的数据取值比它小；
- QU：上四分位数，表示全部观察值中有四分之一的数据取值比它大；
- IQR：四分位间距，是上四分位数QU与下四分位数QL之差，期间包含了全部观察值的一半。

![](https://pic1.zhimg.com/80/v2-052ccb103a1daa584b68e48ae2c6aef0_720w.jpg)

- IQR检测法往往有一个k值，一般为1.5认为$x>QU+k(IQR)||x<QL+k(IQR)$的值为异常值





#### Discussion

- 没有一个单一的异常检测技术是最好的
- 我们需要根据实际情况选择异常分类函数





### 4.7 Handling Outliers

三种常用方式处理异常值

（1）直接通过pandas的条件查询过滤

```python
# Load library
import pandas as pd
# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
# 过滤
print(houses[houses['Bathrooms'] < 20])
```

![image-20220715212809351](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715212809351.png)

（2）方法2，通过np创建一个新的特征，这个特征通过条件判断来判别是不是Outliner

```python
# 方法2，定义一新特征“outliner",然后使用np.where创建条件查询

# Load library
import numpy as np
# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# Show data
print(houses)
```

![image-20220715213450497](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715213450497.png)

方法3：我们可以对某一特征进行数值转换来抑制他的影响

```python
# 方法3：通过数值转换抑制某一特征异常值的影响
# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
# Show data
print(houses)
```

![image-20220715213814307](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715213814307.png)



#### Discussion

- 处理异常值没有一尘不变的规则，需要根据具体情况来选择处理方式
- 我们如何处理异常值应该基于我们的机器学习目标。
- 如果出现异常值，要处理的话需要考虑：为什么是异常值以及最终的目标是什么？
- 不处理异常值也是一种决定
- 如果有异常值那么就不适合标准化，在这种情况下：RobustScaler是更加合理的数据归一化做法。



### 4.8 Discretizating Features

离散化数据

discretizatingExample.py

```python
# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])
# Create binarizer
binarizer = Binarizer(threshold=18)

# Transform feature
print(binarizer.fit_transform(age))

# bin feature
print(np.digitize(age, bins=[20,30,64]))

# Bin feature
print(np.digitize(age, bins=[20,30,64], right=True))
```



![image-20220715215327582](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715215327582.png)

三个实例：

- `binarizer`:对某个阈值进行二分

- `digitize`可以对多个阈值进行划分
- 如果`digitize`的right参数为true那么只有边界右边的值才会被划分到一类种



#### Discussion

- 离散化对于某些问题是个非常有用的方式（原书举例了美国喝酒年龄20和21岁的差异很大的例子）
- 主要有两种方法`binarizer`和`digitize`



### 4.9 Grouping Observations Using Clustering

聚类

划分方法`Kmeans`

```python
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

df = pd.DataFrame(features, columns=["feature_1", "feature_2"])

#  k-means 聚类
clusterer = KMeans(3, random_state=0)

# 过滤数据
clusterer.fit(features)

# 预测分类
df['group'] = clusterer.predict(features)

print(df.head(5))
```

![image-20220715220227238](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715220227238.png)

#### Discussion

- 聚类算法将在19章中详细介绍
- k-means是一种无监督算法，最终得到一个分类的特征
- 查阅资料：[K-Means(K均值聚类算法) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/136842949)
  - 即 K 均值算法，是一种常见的聚类算法。算法会将数据集分为 K 个簇，每个簇使用簇内所有样本均值来表示，将该均值称为“质心”。
  - 容易受初始质心的影响；算法简单，容易实现；算法聚类时，容易产生空簇；算法可能收敛到局部最小值。
  - 距离计算方式是 欧式距离。



### 4.10 Deleteing Observations with Missing Values

删除含空值的observation

```python
# 方法1用numpy
# Load library
import numpy as np
# Create feature matrix
features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])
# Keep only observations that are not (denoted by ~) missing
print(features[~np.isnan(features).any(axis=1)])

# 方法2用pandas
import pandas as pd
# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Remove observations with missing values
print(dataframe.dropna())
```

![image-20220715220853429](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715220853429.png)

#### Discussion

- 大多数机器学习算法无法处理目标和特征数组中的任何缺失值。 
- 最简单的解决方案是删除包含一个或多个缺失值的每个观察值 
- 缺失数据分为三种：
  -  *完全随机丢失（MCAR）* 缺失值的概率与一切无关。 
  -  随机失踪（MAR） 缺失值的概率不是完全随机的，而是取决于其他特征中的信息捕获 
  -  *非随机缺失 (MNAR)* 缺失值的概率不是随机的，取决于我们的特征中未捕获的信息



### 4.11 Imputing Missing Values

预测并且填充丢失的值

inputingExample.py

```fancyimpute```需要安装

- fancyimpute是python的第三方工具包，主要提供了各种[矩阵](https://so.csdn.net/so/search?q=矩阵&spm=1001.2101.3001.7020)计算、填充算法的实现。

```python
# Load libraries
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(n_samples=1000,
                         n_features=2,
                         random_state=1)
# 标准化
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
# 替换第一个值为NAN
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan
# 预测 complete已经被替换为fit_transform
features_knn_imputed = KNN(k=5, verbose=0).fit_transform(standardized_features)
# 比较
print("True Value:", true_value)
print("Imputed Value:", features_knn_imputed[0, 0])


```

![image-20220715223714144](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715223714144.png)

- sklearn中的impute包的方式

```python
#Load library
# impute已经被替代
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

# Create SimpleImputer
mean_imputer = SimpleImputer(strategy="mean")
# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])

```

![image-20220715224036842](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220715224036842.png)

#### Discussion

- 替代值往往有两种主要策略。

  - 我们可以使用机器学习来预测缺失数据的值。为此，我们将具有缺失值的特征视为目标向量，并使用剩余的特征子集来预测缺失值。虽然我们可以使用广泛的机器学习算法来估算值，但流行的选择是 KNN。
  - 更具可扩展性的策略是用一些平均值填充所有缺失值。

- 两种方式各有优缺点

  - KNN 的缺点是，为了知道哪些观测值最接近缺失值，它需要计算缺失值与每个观测值之间的距离。这在较小的数据集中是合理的，但如果数据集有数百万个观测值，很快就会出现问题。
  - 均值的方式往往不像我们使用 KNN 时那样接近真实值

- fancyimpute包查询资料

  | SimpleFill          | 用每列的平均值或中值替换缺失的条目。                         |
  | ------------------- | ------------------------------------------------------------ |
  | KNN                 | 最近邻插补，它使用两行都具有观察数据的特征的均方差对样本进行加权。 |
  | SoftImpute          | 通过 SVD 分解的迭代软阈值完成矩阵[论文笔记 Spectral Regularization Algorithms for Learning Large IncompleteMatrices （soft-impute）_UQI-LIUWJ的博客-CSDN博客](https://blog.csdn.net/qq_40206371/article/details/122416296?spm=1001.2014.3001.5501) |
  | IterativeImpute     | 通过以循环方式将具有缺失值的每个特征建模为其他特征的函数，来估算缺失值。类似于[推荐系统笔记：使用分类模型进行协同过滤_UQI-LIUWJ的博客-CSDN博客](https://blog.csdn.net/qq_40206371/article/details/122088910) |
  | IterativeSVD        | 通过迭代低秩 SVD 分解完成矩阵。类似于 [推荐系统笔记：基于SVD的协同过滤_UQI-LIUWJ的博客-CSDN博客_基于svd的协同过滤](https://blog.csdn.net/qq_40206371/article/details/122143127#:~:text=用另一种-,迭代,-的方法来) |
  | MatrixFactorization | 将不完整矩阵直接分解为低秩 U 和 V，具有每行和每列偏差以及全局偏差。 |
  | BiScaler            | 行/列均值和标准差的迭代估计以获得双重归一化矩阵。 不保证收敛，但在实践中效果很好。 |

- KNN算法

  - [(87条消息) K-近邻算法（KNN)_的博客-CSDN博客_k近邻算法](https://blog.csdn.net/weixin_45884316/article/details/115221211)

  - K近邻（K-Nearest Neighbor, KNN）是一种最经典和最简单的*有监督学习*方法之一。

  - 原理：当对测试样本进行分类时，首先通过扫描训练样本集，找到与该测试样本最相似的个训练样本，根据这个样本的类别进行投票确定测试样本的类别。也可以通过个样本与测试样本的相似程度进行加权投票。如果需要以测试样本对应每类的概率的形式输出，可以通过个样本中不同类别的样本数量分布来进行估计。

  - 三个基本要素

    $ \large k$ 值的选择、距离度量和分类决策规则





## Chapter 5. Handling Categorical Data

### 5.0 Introduction

- 不是根据数量而是根据某些质量来衡量对象通常很有用
- 分类信息通常在数据中表示为向量或字符串列（例如，“Maine”、“Texas”、“Delaware”）。问题是大多数机器学习算法要求输入是数值。
- 我们的目标是进行转换，以正确传达类别中的信息（序数、类别之间的相对间隔等）。在本章中，我们将介绍进行这种转换的技术，以及克服处理分类数据时经常遇到的其他挑战。



### 5.1 Encoding Nominal Categorical Feature

对类别特征编码

encodingCategory.py

```python
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([
    ["Texas"],
    ["California"],
    ["Texas"],
    ["Delaware"],
    ["Texas"]
])

# 创建 one-hot encoder
one_hot = LabelBinarizer()

# one-hot 对feature 编码
print(one_hot.fit_transform(feature))

# 输出各个类别
print(one_hot.classes_)

# 对编码结果解码
print(one_hot.inverse_transform(one_hot.transform(feature)))

# 方法2 使用pandas
# Import library
import pandas as pd

# 从feature中创建dummy编码
print(pd.get_dummies(feature[:, 0]))


multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Create 多类别编码
one_hot_multiclass = MultiLabelBinarizer()
# One-hot encode multiclass feature
print(one_hot_multiclass.fit_transform(multiclass_feature))

# 查看所有类别
print(one_hot_multiclass.classes_)
```

结果：

```
[[0 0 1]
 [1 0 0]
 [0 0 1]
 [0 1 0]
 [0 0 1]]
 
['California' 'Delaware' 'Texas']

['Texas' 'California' 'Texas' 'Delaware' 'Texas']

   California  Delaware  Texas
0           0         0      1
1           1         0      0
2           0         0      1
3           0         1      0
4           0         0      1

[[0 0 0 1 1]
 [1 1 0 0 0]
 [0 0 0 1 1]
 [0 0 1 1 0]
 [1 0 0 0 1]]
 
 ['Alabama' 'California' 'Delware' 'Florida' 'Texas']

```



#### Discussion

- encode的策略并不是为每个类分配一个数值
- encode的正确策略是为每一个类创建一个2元特征；通常这个特征被称为独热编码（one-hot)
  - 查阅资料：[(88条消息) 详细详解One Hot编码-附代码_chary8088的博客-CSDN博客_onehot解码代码](https://blog.csdn.net/chary8088/article/details/79032223
  - One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。
  - One-Hot不适合原本有序的序列，因为会破坏其原本的顺序



### 5.2 Encoding Ordinal Categorical Features

编码有序序列

有效的方式是使用pandas DataFrame中的replace函数

encodeOrdinal.py

```python
# Load library
import pandas as pd

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})
# 创建mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}
# 替换
print(dataframe["Score"].replace(scale_mapper))
```

结果

```
0    1
1    1
2    2
3    2
4    3
Name: Score, dtype: int64
```



#### Discussion

- 顺序序列是我们较为常见的序列，当需要对这些序列进行编码的时候，最好用数值去替换表示顺序的属性
- 当需要表示不同顺序序列之间的距离（程度）时，可以在mapper中凸显出来，例如

​	

```python
scale_mapper = {"Low":1,
				"Medium":2,
				"Barely More Than Medium": 2.1,	# 表示和medium类很贴近
				"High":3}
dataframe["Score"].replace(scale_mapper)
```





### 5.3 Encoding Dictionaries of Features

对字典类型的特征进行encode，通常使用`DictVectorizer`

dictVectorizerExample.py

```python
# Import library
from sklearn.feature_extraction import DictVectorizer
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]
# 创建 dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)
# 默认是生成一个产生稀疏矩阵的dicvectorizer2
dictvectorizer2 = DictVectorizer()
# 把字典转换为特征
features = dictvectorizer.fit_transform(data_dict)
features2 = dictvectorizer2.fit_transform(data_dict)
# 查看 feature matrix
print(features)
print(features2)

# 查看特征的名字 dictvectorizer.get_feature_names即将被遗弃
feature_names = dictvectorizer.get_feature_names_out()
# 查看
print(feature_names)

# 还可以使用pandas
import pandas as pd
# 通过列来表明不同的特征
print(pd.DataFrame(features, columns=feature_names))
```



结果：

```
[[4. 2. 0.]
 [3. 4. 0.]
 [0. 1. 2.]
 [0. 2. 2.]]
  (0, 0)	4.0
  (0, 1)	2.0
  (1, 0)	3.0
  (1, 1)	4.0
  (2, 1)	1.0
  (2, 2)	2.0
  (3, 1)	2.0
  (3, 2)	2.0
['Blue' 'Red' 'Yellow']
   Blue  Red  Yellow
0   4.0  2.0     0.0
1   3.0  4.0     0.0
2   0.0  1.0     2.0
3   0.0  2.0     2.0
```



#### Discussion

- 字典是一个比较常用的数据结构，特别时在自然语言处理中，但是我们最后都需要转换成矩阵的格式
- 字典类型的矩阵非常庞大，如何处理这样的矩阵也是我们的目标（可以使用稀疏矩阵）

- dictvectorizer.get_feature_names即将被遗弃,需要替换成dictvectorizer.get_feature_names_out()



### 5.4 Imputing Missing Class Values

填充丢失的类别值

最常用的方法仍然是KNN

imputing.py

```python
# load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0, 2.10, 1.45],
            [1, 1.18, 1.33],
            [0, 1.22, 1.27],
            [1, -0.21, -1.19]])

X_with_nan = np.array([[np.nan, 0.87, 1.31],
                      [np.nan, -0.67, -0.22]])

# 训练 KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:, 0])

# 预测丢失值
imputed_values = trained_model.predict(X_with_nan[:, 1:])

# 连接丢失的列和他们原来其他列
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# 连接原来的矩阵和有丢失列的矩阵
print(np.vstack((X_with_imputed, X)))

# 方法2 用最常用的值去替换缺失值
# impute已经被替代
from sklearn.impute import SimpleImputer
# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
imputer = SimpleImputer(strategy='most_frequent')
print(imputer.fit_transform(X_complete))
```



结果

```
[[ 0.    0.87  1.31]
 [ 1.   -0.67 -0.22]
 [ 0.    2.1   1.45]
 [ 1.    1.18  1.33]
 [ 0.    1.22  1.27]
 [ 1.   -0.21 -1.19]]
 
 [[ 0.    0.87  1.31]
 [ 0.   -0.67 -0.22]
 [ 0.    2.1   1.45]
 [ 1.    1.18  1.33]
 [ 0.    1.22  1.27]
 [ 1.   -0.21 -1.19]]
```



#### Discussion

- KNN是最常用的方法，测量值比较精准，但是在大数据面前表现比较差
- 用最常见的特征值没有KNN准确，但是它在大数据情况时可拓展性比KNN好，而且没有KNN那么复杂



### 5.5 Handling Imbalanced Classes

我们如果遇到极度不平衡的类的目标向量，如何解决

- 作者认为首先应该去搜集更多的数据

- 如果搜集的数据已经足够多的话，应该更改评估模型的指标

- 再然后如果还不能解决就要使用内置的类权重参数、下采样或者上采样

- 评估模型指标在之后的章节会介绍，这一节关注类权重参数、下采样和上采样的方法

  handleImbalance.py

```python
# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
# 加载莺尾花数据集
iris = load_iris()
# 创建特征矩阵
features = iris.data
# 创建目标向量
target = iris.target
# 移除前40个元素
features = features[40:,:]
target = target[40:]
# 将类0和非类0的observation分为两类
target = np.where((target == 0), 0, 1)
# 打印
print(target)

# 第一种处理，提供类权重

# 创建权重
weights = {0: .9, 1: 0.1}
# 根据权重创建随机森岭
print(RandomForestClassifier(class_weight=weights))

# Train a random forest with balanced class weights
print(RandomForestClassifier(class_weight="balanced"))

# 处理方式2 进行下采样
# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]
# 读取类的大小
n_class0 = len(i_class0)
n_class1 = len(i_class1)
# For every observation of class 0, randomly sample
# 随机取样，使得class1 Observation数量和class0一样多
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
# 连接
# 下采样
print(np.hstack((target[i_class0], target[i_class1_downsampled])))
print(np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5])

# 处理方式3 上采样
# 对于class0来说,随机创建样本直至数量和class1一样
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# 目标向量
print(np.concatenate((target[i_class0_upsampled], target[i_class1])))
# 特征矩阵
print(np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5])
```



```python
[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]


RandomForestClassifier(class_weight={0: 0.9, 1: 0.1})
RandomForestClassifier(class_weight='balanced')


[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
[[5.  3.5 1.3 0.3]
 [4.5 2.3 1.3 0.3]
 [4.4 3.2 1.3 0.2]
 [5.  3.5 1.6 0.6]
 [5.1 3.8 1.9 0.4]]


[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
[[5.1 3.8 1.9 0.4]
 [5.1 3.8 1.6 0.2]
 [5.3 3.7 1.5 0.2]
 [5.3 3.7 1.5 0.2]
 [4.5 2.3 1.3 0.3]]


```



#### Discussion

- 在现实世界中，不平衡的课程无处不在——大多数访问者不会点击购买按钮，幸好许多类型的癌症很少见。出于这个原因，处理不平衡类是机器学习中的常见活动。
- 策略优先级从高到低分别是
  - 收集更多的观察结果——尤其是少数群体的观察结果。
  - 使用更适合不平衡类的模型评估指标。我们在后面的章节中讨论的一些更好的指标是混淆矩阵、精度、召回、F1 分数和 ROC 曲线。
  - 使用包含在某些模型实现中的类权重参数。这使我们可以让算法针对不平衡的类进行调整。幸运的是，许多 scikit-learn 分类器都有一个 class_weight 参数，使其成为一个不错的选择。
  - 下采样和上采样。通常我们应该尝试两者来查看哪个产生更好的结果。





## Chapter 6. Handling Text

### 6.0 Introduction

- 在本章中，我们将介绍将文本转换为信息丰富的特征的策略。
- 因为处理文本信息的方法过多，本章将只能重点介绍。
- 本章介绍的这些常规技术是非常有价值的预处理工具



### 6.1 Cleaning Text

对非结构性文本进行一些基本的清理

常用的函数：

`strip`

`replace`

`split`

clean.py

```python
# 正则表达式模块
import re

# 新建一段文本
text_data = [" Interrobang. By Aishwarya Henriette ",
             "Parking And Going. By Karl Gautier",
             " Today Is The night. By Jarek Prakash "]
# 除去始末空格
strip_whitespace = [string.strip() for string in text_data]
# Show text
print(strip_whitespace)

# 除去.
remove_periods = [string.replace(".", "") for string in strip_whitespace]
# Show text
print(remove_periods)


# Create function
def capitalizer(string: str) -> str:
    return string.upper()


# 全部变成大写
print([capitalizer(string) for string in remove_periods])

# 新建一个正则表达式匹配函数
def replace_letters_with_upper_x(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)


# 运用function
print([replace_letters_with_upper_x(string) for string in remove_periods])
```

![image-20220717115227664](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717115227664.png)



#### Discussion

- 大多数文本数据都需要在我们使用它来构建功能之前进行清理。 
- 大多数基本的文本清理都可以使用 Python 的标准字符串操作来完成。
- 可以自定义函数完成处理



### 6.2 Parsing and Cleaning HTML

- 处理HTML文本
- 使用Beautiful Soup库



cleanHtml.py

- 需要安装两个库

- Beautiful Soup 4

  官网：[Beautiful Soup Documentation — Beautiful Soup 4.4.0 documentation (beautiful-soup-4.readthedocs.io)](https://beautiful-soup-4.readthedocs.io/en/latest/)

  [Beautiful Soup 中文文档](https://beautifulsoup.cn/)

  教程[BeautifulSoup详细使用教程！你学会了吗？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/59822990)

  ```
  conda install bs4
  ```

- lxml

  [Python lxml库的安装和使用 (biancheng.net)](http://c.biancheng.net/python_spider/lxml.html)

  ```
  conda install lxml
  ```

  

  代码：

  ```
  # Load library
  from bs4 import BeautifulSoup
  
  # 创建一些html文本
  html = """
  <div class='full_name'><span style='font-weight:bold'>
  Masego</span> Azra</div>"
  """
  # 转换 html
  soup = BeautifulSoup(html, "lxml")
  # 查找div，寻找class为fullname的标签，获得它的文本属性
  print(soup.find("div", {"class": "full_name"}).text)
  ```

  ![image-20220717141910538](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717141910538.png)

#### Discussion

- Beautiful Soup 是一个强大的 Python 库，专为抓取 HTML 而设计。

- 支持原生和第三方解析器

  | 解析器           | 使用方法                                                     | 优势                                                  | 劣势                                            |
  | ---------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------- |
  | Python标准库     | BeautifulSoup(markup, “html.parser”)                         | Python的内置标准库执行速度适中文档容错能力强          | Python 2.7.3 or 3.2.2)前 的版本中文档容错能力差 |
  | lxml HTML 解析器 | BeautifulSoup(markup, “lxml”)                                | 速度快文档容错能力强                                  | 需要安装C语言库                                 |
  | lxml XML 解析器  | BeautifulSoup(markup, [“lxml”, “xml”])BeautifulSoup(markup, “xml”) | 速度快唯一支持XML的解析器                             | 需要安装C语言库                                 |
  | html5lib         | BeautifulSoup(markup, “html5lib”)                            | 最好的容错性以浏览器的方式解析文档生成HTML5格式的文档 | 速度慢不依赖其他库                              |

- Beautiful Soup四大标签

  - Tag 
  -  NavigableString 
  -  BeautifulSoup 
  -  Comment

- 支持遍历文档树，查找文档树以及CSS选择器三个主要操作





### 6.3 Removing Punctuation

移除标点

removingPunctuation.py

```python
# Load libraries
import unicodedata
import sys

# 文本
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']
# 创建一个字典，用于处理标点
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
# For each string, 移除标点
print([string.translate(punctuation) for string in text_data])
```



![image-20220717143110998](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717143110998.png)

#### Discussion

- translate 是一种 Python 方法，因其超快的速度而广受欢迎。
- 在该例中，我们创建一个字典，用unicodedata.category方法查找出所有unicode中属于标点符号的字符，并把它映射到None上，最后再用translate方法把标点符号全部转换为空值



### 6.4 Tokenizing Text

- 把文本分解为单个单词

- NLTK——强大的python自然语言工具包，具有强大的文本集操作

- nltk包需要安装

  ```
  conda install nltk
  ```

- nltk需要下载数据包

  ```python
  # 主动下载
  import nltk
  nltk.download('punkt')
  
  # 离线下载
  # 下载到指定的文件夹里
  # http://www.nltk.org/nltk_data/
  ```

  

NTLKExample.py

```python
# Load library

from nltk.tokenize import word_tokenize
# Create text
string = "The science of today is the technology of tomorrow"
# 分词
print(word_tokenize(string))

# Load library
from nltk.tokenize import sent_tokenize
# Create text
string = "The science of today is the technology of tomorrow. Tomorrow is today."
# 分句子
print(sent_tokenize(string))
```

主动下载+结果

![image-20220717145912597](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717145912597.png)



#### Discussion

- 官网：[NLTK :: Natural Language Toolkit](https://www.nltk.org/)
- 首先nltk的运行是需要它自身的开源**语料库**、**词库**、**标记库**进行的
- 提供非常广泛的功能，例如词性分析，词性还原、还可以进行朴素贝叶斯分类



### 6.5 Removing Stop Words

移除去stop words(信息量极少的次，例如‘I' 'am'等)

仍然需要NLTK库

stopWorkds.py

```
# Load library

from nltk.corpus import stopwords

# 需要下载
import nltk
nltk.download('stopwords')
# Create word tokens
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']
# Load stop words
stop_words = stopwords.words('english')
# Remove stop words
print([word for word in tokenized_words if word not in stop_words])
```

![image-20220717151128933](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717151128933.png)

#### Discussion

- 虽然“stop words”可以指代我们想要在处理之前删除的任何一组词，但该术语通常指的是极其常见的词，它们本身包含的信息价值很少。 NLTK 有一个常见停用词列表，我们可以使用这些停用词在我们的标记词中查找和删除停用词：

```
# Show stop words
stop_words[:5]
```

['i', 'me', 'my', 'myself', 'we']

- 需要注意ntlk中的stop words都是小写的



### 6.6 Stemming Words

- 把单词转换成原型
- 需要使用nltk中的`PorterStemmer`

stemmingWords.py

```python
# Load library
from nltk.stem.porter import PorterStemmer
# Create word tokens
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# 创建 stemmer
porter = PorterStemmer()
# 运用 stemmer
print([porter.stem(word) for word in tokenized_words])
```

![image-20220717151651206](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717151651206.png)

#### Discussion

- 虽然将句子中的词转换为词干可读性较差，但是更容易将句子观察比较

- PorterStemmer使用了现在非常常用的Porter算法来删除一些常见的前缀和后缀生成stemming

#### Porter算法：[(89条消息) Porter Algorithm ---------词干提取算法_noobzc1的博客-CSDN博客_词干提取算法](https://blog.csdn.net/noobzc1/article/details/8902881#)（网上有python实现，但是最经典的版本是java实现的）

- 定义一个类	

  ```java
  class Stemmer
  {  private char[] b;
     private int i,     /* b中的元素位置（偏移量） */
                 i_end, /* 要抽取词干单词的结束位置 */
                 j, k;
     private static final int INC = 50;
                       /* 随着b的大小增加数组要增长的长度（防止溢出） */
     public Stemmer()
     {  b = new char[INC];
        i = 0;
        i_end = 0;
     }
  }		
  ```

  

- 字符串处理add

  ```java
  /**
  * 增加一个字符到要存放待处理的单词的数组。添加完字符时，
  * 可以调用stem(void)方法来进行抽取词干的工作。
  */
  public void add(char ch)
  {  if (i == b.length)
     {  char[] new_b = new char[i+INC];
        for (int c = 0; c < i; c++) new_b[c] = b[c];
        b = new_b;
     }
     b[i++] = ch;
  }
   
  /** 增加wLen长度的字符数组到存放待处理的单词的数组b。
  */
  public void add(char[] w, int wLen)
  {  if (i+wLen >= b.length)
     {  char[] new_b = new char[i+wLen+INC];
        for (int c = 0; c < i; c++) new_b[c] = b[c];
        b = new_b;
     }
     for (int c = 0; c < wLen; c++) b[i++] = w[c];
  }
  ```

  

- 一系列辅助函数

  总结下来就是判断发音类（辅音和元音）和操作类（setto和r），比较复杂的函数应该是m()返回辅音序列的个数和cvc(i)处理e结尾的单词

  - **cons(i)**：参数i：int型；返回值bool型。当i为辅音时，返回真；否则为假。

  - **m（）**

    ：返回值：int型。表示单词b介于0和j之间辅音序列的个度。现假设c代表辅音序列，而v代表元音序列。<..>表示任意存在。于是有如下定义；

    - <c><v>      结果为 0
    - <c>vc<v>    结果为 1
    - <c>vcvc<v>   结果为 2
    - <c>vcvcvc<v> 结果为 3
    - ....

  - **vowelinstem()**：返回值：bool型。从名字就可以看得出来，表示单词b介于0到i之间是否存在元音。

  - **doublec(j)**：参数j：int型；返回值bool型。这个函数用来表示在j和j-1位置上的两个字符是否是相同的辅音。

  - **cvc(i)**：参数i：int型；返回值bool型。对于i，i-1，i-2位置上的字符，它们是“辅音-元音-辅音”的形式，并且对于第二个辅音，它不能为w、x、y中的一个。这个函数用来处理以e结尾的短单词。比如说cav(e)，lov(e)，hop(e)，crim(e)。但是像snow，box，tray就辅符合条件。

  - **ends(s)**：参数：String；返回值：bool型。顾名思义，判断b是否以s结尾。

  - **setto(s)**：参数：String；void类型。把b在(j+1)...k位置上的字符设为s，同时，调整k的大小。

  - **r(s)**：参数：String；void类型。在m()>0的情况下，调用setto(s)。

  ```java
  // cons(i) 为真 <=> b[i] 是一个辅音
  private final boolean cons(int i)
  {  switch (b[i])
     {  case 'a': case 'e': case 'i': case 'o': case 'u': return false; //aeiou
        case 'y': return (i==0) ? true : !cons(i-1);
                  //y开头，为辅；否则看i-1位，如果i-1位为辅，y为元，反之亦然。
        default: return true;
     }
  }
   
  // m() 用来计算在0和j之间辅音序列的个数。 见上面的说明。 */
  private final int m()
  {  int n = 0; //辅音序列的个数，初始化
     int i = 0; //偏移量
     while(true)
     {  if (i > j) return n; //如果超出最大偏移量，直接返回n
        if (! cons(i)) break; //如果是元音，中断
        i++; //辅音移一位，直到元音的位置
     }
     i++; //移完辅音，从元音的第一个字符开始
     while(true)//循环计算vc的个数
     {  while(true) //循环判断v
        {  if (i > j) return n;
           if (cons(i)) break; //出现辅音则终止循环
              i++;
        }
        i++;
        n++;
        while(true) //循环判断c
        {  if (i > j) return n;
           if (! cons(i)) break;
           i++;
        }
        i++;
      }
  }
   
  // vowelinstem() 为真 <=> 0,...j 包含一个元音
  private final boolean vowelinstem()
  {  int i; for (i = 0; i <= j; i++) if (! cons(i)) return true;
     return false;
  }
   
  // doublec(j) 为真 <=> j,(j-1) 包含两个一样的辅音
  private final boolean doublec(int j)
  {  if (j < 1) return false;
     if (b[j] != b[j-1]) return false;
     return cons(j);
  }
   
  /* cvc(i) is 为真 <=> i-2,i-1,i 有形式： 辅音 - 元音 - 辅音
     并且第二个c不是 w,x 或者 y. 这个用来处理以e结尾的短单词。 e.g.
   
     cav(e), lov(e), hop(e), crim(e), 但不是
     snow, box, tray.
   
  */
  private final boolean cvc(int i)
  {  if (i < 2 || !cons(i) || cons(i-1) || !cons(i-2)) return false;
     {  int ch = b[i];
           if (ch == 'w' || ch == 'x' || ch == 'y') return false;
     }
        return true;
  }
   
  private final boolean ends(String s)
  {  int l = s.length();
     int o = k-l+1;
     if (o < 0) return false;
     for (int i = 0; i < l; i++) if (b[o+i] != s.charAt(i)) return false;
     j = k-l;
     return true;
  }
   
  // setto(s) 设置 (j+1),...k 到s字符串上的字符, 并且调整k值
  private final void setto(String s)
  {  int l = s.length();
     int o = j+1;
     for (int i = 0; i < l; i++) b[o+i] = s.charAt(i);
     k = j+l;
  }
   
  private final void r(String s) { if (m() > 0) setto(s); }
  ```

  

- 然后正式进入六步操作的工作

  - 第一步是否为ed或着ing结尾

    可以明显看到麻烦在于分类讨论，有许多诸如sses或者ies这样的结尾很难判断

  ```java
  /* step1() 处理复数，ed或者ing结束的单词。比如：
   
        caresses  ->  caress
        ponies    ->  poni
        ties      ->  ti
        caress    ->  caress
        cats      ->  cat
   
        feed      ->  feed
        agreed    ->  agree
        disabled  ->  disable
   
        matting   ->  mat
        mating    ->  mate
        meeting   ->  meet
        milling   ->  mill
        messing   ->  mess
   
        meetings  ->  meet
  */
   
  private final void step1()
  {  if (b[k] == 's')
     {  if (ends("sses")) k -= 2; //以“sses结尾”
        else if (ends("ies")) setto("i"); //以ies结尾，置为i
        else if (b[k-1] != 's') k--; //两个s结尾不处理
     }
     if (ends("eed")) { if (m() > 0) k--; } //以“eed”结尾，当m>0时，左移一位
     else if ((ends("ed") || ends("ing")) && vowelinstem())
     {  k = j;
        if (ends("at")) setto("ate"); else
        if (ends("bl")) setto("ble"); else
        if (ends("iz")) setto("ize"); else
        if (doublec(k))//如果有两个相同辅音
        {  k--;
           {  int ch = b[k];
              if (ch == 'l' || ch == 's' || ch == 'z') k++;
           }
        }
        else if (m() == 1 && cvc(k)) setto("e");
    }
  }
  ```

  

  - 第二步 如果含有元音，并且以y结尾将y改成i

    ```java
    private final void step2() { if (ends("y") && vowelinstem()) b[k] = 'i'; }
    ```

    

  - 第三步 将双后缀的单词映射为单后缀。 和第一步一样需要分类讨论，有很多英语上的特殊情况例如

    所以只能一个一个进行判断，但是实际上就是枚举所有类别的双后缀然后转换成原来的模式

    ```java
    /* step3() 将双后缀的单词映射为单后缀。 所以 -ization ( = -ize 加上
       -ation) 被映射到 -ize 等等。 注意在去除后缀之前必须确保
       m() > 0. */
    private final void step3() { if (k == 0) return;  switch (b[k-1])
    {
        case 'a': if (ends("ational")) { r("ate"); break; }
                  if (ends("tional")) { r("tion"); break; }
                  break;
        case 'c': if (ends("enci")) { r("ence"); break; }
                  if (ends("anci")) { r("ance"); break; }
                  break;
        case 'e': if (ends("izer")) { r("ize"); break; }
                  break;
        case 'l': if (ends("bli")) { r("ble"); break; }
                  if (ends("alli")) { r("al"); break; }
                  if (ends("entli")) { r("ent"); break; }
                  if (ends("eli")) { r("e"); break; }
                  if (ends("ousli")) { r("ous"); break; }
                  break;
        case 'o': if (ends("ization")) { r("ize"); break; }
                  if (ends("ation")) { r("ate"); break; }
                  if (ends("ator")) { r("ate"); break; }
                  break;
        case 's': if (ends("alism")) { r("al"); break; }
                  if (ends("iveness")) { r("ive"); break; }
                  if (ends("fulness")) { r("ful"); break; }
                  if (ends("ousness")) { r("ous"); break; }
                  break;
        case 't': if (ends("aliti")) { r("al"); break; }
                  if (ends("iviti")) { r("ive"); break; }
                  if (ends("biliti")) { r("ble"); break; }
                  break;
        case 'g': if (ends("logi")) { r("log"); break; }
    } }
    ```

  -  第四步，处理 -ic-，-full，-ness等等后缀。和步骤3有着类似的处理。 也是分类讨论然后替换

    ```java
    private final void step4() { switch (b[k])
    {
        case 'e': if (ends("icate")) { r("ic"); break; }
                  if (ends("ative")) { r(""); break; }
                  if (ends("alize")) { r("al"); break; }
                  break;
        case 'i': if (ends("iciti")) { r("ic"); break; }
                  break;
        case 'l': if (ends("ical")) { r("ic"); break; }
                  if (ends("ful")) { r(""); break; }
                  break;
        case 's': if (ends("ness")) { r(""); break; }
                  break;
    } }
    ```

    

  - 第五步就是根据m（）的统计情况，处理<c>vcvc<v>的情况

    ```java
    private final void step5()
    {   if (k == 0) return;  switch (b[k-1])
        {  case 'a': if (ends("al")) break; return;
           case 'c': if (ends("ance")) break;
                     if (ends("ence")) break; return;
           case 'e': if (ends("er")) break; return;
           case 'i': if (ends("ic")) break; return;
           case 'l': if (ends("able")) break;
                     if (ends("ible")) break; return;
           case 'n': if (ends("ant")) break;
                     if (ends("ement")) break;
                     if (ends("ment")) break;
                     /* element etc. not stripped before the m */
                     if (ends("ent")) break; return;
           case 'o': if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
                                     /* j >= 0 fixes Bug 2 */
                     if (ends("ou")) break; return;
                     /* takes care of -ous */
           case 's': if (ends("ism")) break; return;
           case 't': if (ends("ate")) break;
                     if (ends("iti")) break; return;
           case 'u': if (ends("ous")) break; return;
           case 'v': if (ends("ive")) break; return;
           case 'z': if (ends("ize")) break; return;
           default: return;
        }
        if (m() > 1) k = j;//调用对k赋值
    }
    ```

     

  - 第6步很好理解 除去末尾冗余的e

    ```java
    private final void step6()
    {  j = k;
       if (b[k] == 'e')
       {  int a = m();
          if (a > 1 || a == 1 && !cvc(k-1)) k--;
       }
       if (b[k] == 'l' && doublec(k) && m() > 1) k--;
    }
    ```

    

### 6.7 Tagging Parts of Speech

标记词性

tagging.py

```python
# Load libraries
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
# 下载
# import nltk
# nltk.download('averaged_perceptron_tagger')
# Create text


text_data = "Chris loved outdoor running"
# 使用训练好的模型处理
text_tagged = pos_tag(word_tokenize(text_data))
# Show parts of speech
print(text_tagged)

# Filter words
print([word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']])

# 用独热编码将词性统计转化为特征矩阵
# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]
# Create list
tagged_tweets = []
# 标记每个词的词性
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
	tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot 编码
one_hot_multi = MultiLabelBinarizer()
print(one_hot_multi.fit_transform(tagged_tweets))

print(one_hot_multi.classes_)
```

![image-20220717155353522](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717155353522.png)

![image-20220717155647789](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717155647789.png)

#### Discussion

- 如果是非专业术语语句，使用 NLTK 的预训练词性标注器是最简单的方法

- Brown Corpus是NTLK使用的语料库

- NLTK uses the Penn Treebank parts for speech tags, some examples:

  

  | tag  | Parts of Speech                    |
  | ---- | ---------------------------------- |
  | NNP  | Proper noun, singular              |
  | NN   | Noun, singular or mass             |
  | RB   | Adverb                             |
  | VBD  | Verb, past tense                   |
  | VBG  | Verb, gerund or present participle |
  | JJ   | Adjective                          |
  | PRP  | Personal  pronoun                  |

- 这里我们使用一个退避 n-gram 标注器，其中 n 是我们在预测词的词性标签时考虑的先前词的数量。首先我们使用 TrigramTagger 考虑前面两个词；如果两个单词不存在，我们“退后”并使用 BigramTagger 考虑前一个单词的标签，最后如果失败，我们只使用 UnigramTagger 查看单词本身。

  ```python
  # Load library
  from nltk.corpus import brown
  from nltk.tag import UnigramTagger
  from nltk.tag import BigramTagger
  from nltk.tag import TrigramTagger
  # Get some text from the Brown Corpus, broken into sentences
  sentences = brown.tagged_sents(categories='news')
  # Split into 4000 sentences for training and 623 for testing
  train = sentences[:4000]
  test = sentences[4000:]
  # backoff tagger
  unigram = UnigramTagger(train)
  bigram = BigramTagger(train, backoff=unigram)
  trigram = TrigramTagger(train, backoff=bigram)
  # 准确率 :0.8179229731754832
  print(trigram.evaluate(test))
  ```

  

### 6.8 Encoding Text as a Bag of Words

统计特定的词出现的次数

使用`scikit-learn’s CountVectorizer`

countExample.py

```python
# Load library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# 创建一个特征矩阵包含计数信息
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# 打印
print(bag_of_words)

print(bag_of_words.toarray())

# 展示特征的name get_feature_names即将废弃
print(count.get_feature_names_out())
```

![image-20220717160733410](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717160733410.png)

#### Discussion

- Bag-of-words models是文本转换成feature最常用的模型
- 大多数bag-of-words的矩阵都是稀疏矩阵，所以CountVectorizer 的一个很好的特性是默认情况下输出是一个稀疏矩阵
- CountVectorizer 带有许多有用的参数，可以轻松创建词袋特征矩阵。首先，虽然默认情况下每个特征都是一个单词，但不一定是这样。相反，我们可以将每个特征设置为两个单词（称为 2-gram）甚至三个单词（3-gram）的组合。

- ngram_range 设置我们的 n-gram 的最小和最大大小。 例如，(2,3) 将返回所有 2-gram 和 3-gram。 其次，我们可以使用内置列表或自定义列表的 stop_words 轻松删除低信息填充词。 最后，我们可以使用词汇将我们想要考虑的单词或短语限制在某个单词列表中。 例如，我们可以为仅出现的国家名称创建一个词袋特征矩阵：

  ```python
  # Create feature matrix with arguments
  count_2gram = CountVectorizer(ngram_range=(1, 2),
                                stop_words="english",
                                vocabulary=['brazil'])
  bag = count_2gram.fit_transform(text_data)
  # View feature matrix
  print(bag.toarray())
  # View the 1-grams and 2-grams
  print(count_2gram.vocabulary_)
  ```

  ![image-20220717161421563](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717161421563.png)

- N-Gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。

  [自然语言处理中N-Gram模型介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32829048)





### 6.9 Weighting Word Importance

- 给单词加权
- 使用词频-逆文档频率 (tf-idf) 比较文档（推文、电影评论、演讲稿等）中单词的频率与所有其他文档中单词的频率。 scikit-learn 使用 TfidfVectorizer 使这变得简单：

weightWords.py

```python
# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# 创建 the tf-idf 特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# 展示这个特征矩阵
print(feature_matrix)

# 转换为一般数组
print(feature_matrix.toarray())
# 特征名称
print(tfidf.vocabulary_)
```



![image-20220717162048935](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220717162048935.png)



#### Discussion

- 一个词在文档中出现的越多，它对该文档的重要性就越大。 例如，如果经济一词频繁出现，则证明该文件可能是关于经济的。 我们称之为术语频率 (tf)。
- 如果一个词出现在许多文档中，那么它对任何单个文档的重要性都可能降低。 例如，如果某些文本数据中的每个文档都包含后面的单词，那么它可能是一个不重要的单词。 我们称此文档频率 (df)。
- 通过结合这两个统计数据，我们可以为每个单词分配一个分数，表示该单词在文档中的重要性。 具体来说，我们将 tf 乘以文档频率 (idf) 的倒数：

​	$$tf-idf(t, d) = tf(t,d) * idf(t)$$

- tf 和 idf 的计算方式有很多变化。 在 scikit-learn 中，tf 只是单词在文档中出现的次数，idf 的计算公式为：

  $$idf(t) = log(\frac{1 + n_d}{1 + df(d, t}) +1$$

​	其中 nd 是文档数，df(d,t) 是术语，t 的文档频率（即，该术语出现的文档数）。默认情况下，scikit-learn 	然后使用欧几里得范数（L2 范数）对 tf-idf 向量进行归一化。结果值越高，单词对文档越重要





## Chapter 7. Handling Dates and Times

### 7.0 Introduction

- 日期和时间是机器学习中常见要处理的类型
-  `pandas` 库中的时间序列工具，它集中了许多其他库的功能。



### 7.1 Converting Strings to Dates

- 转换字符串为时间
- `pandas’ to_datetime`

strToDate.py

```python
import numpy as np
import pandas as pd

date_strings = np.array([
    '03-04-2005 11:35 PM',
    '23-05-2010 12:01 AM',
    '04-09-2009 09:09 PM'
])

# 转换 datetimes
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings])


# coerce设置时 不会raise error 而会输出NaT
date_strings = np.append(date_strings,["13-13-3333 25:61 PP"])
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p', errors='coerce') for date in date_strings])
```

![image-20220718114919034](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718114919034.png)

#### Discussion

- `pandas’ to_datetime`可以将字符串转换为时间

- 可以使用 format 参数来指定字符串的确切格式。

- 创建的日期格式

  | Code | Description                       | Example |
  | ---- | --------------------------------- | ------- |
  | %Y   | Full year                         | 2001    |
  | %m   | Month w/ zero padding             | 04      |
  | %d   | Day of the month w/ zero padding  | 09      |
  | %I   | Hour (12hr clock) w/ zero padding | 02      |
  | %p   | AM or PM                          | AM      |
  | %M   | Minute w/ zero padding            | 05      |
  | %S   | Second w/ zero padding            | 09      |



### 7.2 Handling Time Zones

- 处理时区
- 如果未指定，pandas 对象没有时区。 但是，我们可以在创建过程中使用 `tz `添加时区：

timeZone.py

```python

import pandas as pd
print(pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London'))

date = pd.Timestamp('2017-05-01 06:00:00')
# 设置 time zone
date_in_london = date.tz_localize('Europe/London')
# 打印
print(date_in_london)

# 更改时区 time zone
print(date_in_london.tz_convert('Africa/Abidjan'))

# 通过pd一次性创建3个时间段
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
# 设置时区
print(dates.dt.tz_localize('Africa/Abidjan'))
```

![image-20220718203729330](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718203729330.png)

#### Discussion

- 建议使用 pytz 库的字符串
- 导入 all_timezones 查看所有用于表示时区的字符串：

```python
# Load library
from pytz import all_timezones
# 展示前两个时区
print(all_timezones[0:2])
```

![image-20220718204818920](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718204818920.png)



### 7.3 Selecting Dates and Times

- 从一组日期中选择出特定的日期
- pandas dataFrame的切片访问

selectTime.py

```python
# Load library
import pandas as pd
# 创建空的DataFrame
dataframe = pd.DataFrame()
# 填充数据
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
# Select 
print(dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
(dataframe['date'] <= '2002-1-1 04:00:00')])

```

![image-20220718205738464](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718205738464.png)

#### Discussion

- bool条件值或者切片
- 作者认为在大数据时应该使用切片，而数据量较小的时候使用bool值访问更加合适



### 7.4 Breaking Up Date Data into Multiple Features

- 将日期拆解成多个特征
- `pandas Series.dt`

seriesDt.py

```python
# Load library
import pandas as pd
# 空的dataFrame
dataframe = pd.DataFrame()
# 创建 150个数据
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')
# 使用dt拆解成年月日小时分钟
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute
# Show
print(dataframe.head(3))
```

![image-20220718210303226](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718210303226.png)

#### Discussion

- 分解时间作者认为有时很有用比如只考察某样事物一年间每月的变化



### 7.5 Calculating the Difference Between Dates

- 想要计算时间差
- `pandas`

timestamp.py

```python
# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# 创建两个时间
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
# 计算持续时间
print(dataframe['Left'] - dataframe['Arrived'])

# 计算持续时间
print(pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived'])))
```

![image-20220718211219675](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718211219675.png)

#### Discussion

- 计算时间差值很有用
- pandas中的TimeDelta使得计算时间差值很简单

```python
# 计算时间差
timedelta = pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
print(timedelta)
```

![image-20220718211626892](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718211626892.png)



### 7.6 Encoding Days of the Week

- 对星期编码
- `Series.dt`

weekdays.py

```python
# Load library
import pandas as pd
# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))
# Show days of the week
print(dates.dt.weekday_name)
# Show days of the week
print(dates.dt.weekday)
```

报错

![image-20220718212047131](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718212047131.png)

根据查阅资料weekday_name已经被废弃，改为day_name

```python
# Load library
import pandas as pd
# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))
# 星期几
print(dates.dt.day_name())
# 数字编号
print(dates.dt.weekday)
```

![image-20220718213120073](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220718213120073.png)

#### Discussion

- 对于分析星期类的问题，pandas的转换会很有帮助





### 7.7 Creating a Lagged Feature

- 创建滞后特征
- 使用pandas库的`shift`

```python
# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
# 创建一个新的特征前移一天
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)
# 展现dataframe
print(dataframe)
```

![image-20220718213839590](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718213839590.png)

#### Discussion

- 什么是**Lagged Feature**?
  - 数据通常基于有规律的间隔时间段（例如，每天、每小时、每三个小时），我们有兴趣使用过去的值进行预测,而这些过去的特征就可以称作**滞后特征（Lagged Feature）**

- 在我们的示例中因为往前移了一天，第一行没有数据，所以是NaN；



###  7.8 Using Rolling Time Windows 

- 计算这些流动时间的统计数据

statistic.py

```python
# Load library
import pandas as pd
# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# 设置index
dataframe = pd.DataFrame(index=time_index)
# 新建一个特征
dataframe["Stock_Price"] = [1,2,3,4,5]
# 窗口大小为2，求出窗口时间中的平均值
print(dataframe.rolling(window=2).mean())
```

![image-20220718214653455](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718214653455.png)

#### Discussion

- 滚动时间窗口包含连续的一些时间，然后一次次的移动
- 每次可以求出窗口内的统计数据

### 7.9 Handling Missing Data in Time Series

- 处理丢失数据
- 类似于之前处理丢失数据的方式
- 我们对于时间数据还可以使用插值法

handlingMissing.py

```python
# Load libraries
import pandas as pd
import numpy as np

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# set index
dataframe = pd.DataFrame(index=time_index)
# 创建一组有缺失的数据
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]
# 插入缺失值
print(dataframe.interpolate())

# 前值填充
print(dataframe.ffill())

# 后值填充
print(dataframe.bfill())

```

![image-20220718215920807](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718215920807.png)

#### Discussion

- 插值是一种填补由缺失值引起的空白的技术，实际上是在与空白接壤的已知值之间绘制一条直线或曲线，并使用该直线或曲线来预测合理的值。

  - 我们的解决方案中，两个缺失值的差距以 2.0 和 5.0 为界。通过拟合从 2.0 到 5.0 的线，我们可以对介于 3.0 和 4.0 之间的两个缺失值做出合理的猜测。

  -  如果我们认为两个已知点之间的线是非线性的，我们可以使用 interpolate 的方法来指定插值方法：

    ```python
    # 非线性 结果见下图
    print(dataframe.interpolate(method="quadratic"))
    ```

    

- 在某些情况下，我们的缺失值差距很大，并且不想在整个差距中插入值。在这些情况下，我们可以使用 limit 来限制插值的数量，使用 limit_direction 来设置是否从间隙之前的最后一个已知值向前插值，

```python
# 限制
print(dataframe.interpolate(limit=1, limit_direction="forward"))
```

![image-20220718221132787](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220718221132787.png)





## Chapter 8. Handling Images

### 8.0 Introduction

- 我们将机器学习应用于图像之前，我们通常首先需要将原始图像转换为我们的学习算法可用的特征。
- 要处理图像，我们将使用开源计算机视觉库 (OpenCV)。

​	

```
conda install -c https://conda.anaconda.org/menpo opencv
```

```
import cv2
print(cv2.__version__)
```

4.5.2



### 8.1 Loading Images

- 加载一张图片

- `cv2.imread`

  ```python
  # Load library
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  # 加载图片
  image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
  
  # 使用plt展示图片
  plt.imshow(image, cmap="gray"), plt.axis("off")
  plt.show()
  
  ```

  结果：

  ![image-20220719095214902](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719095214902.png)

#### Discussion

- 从根本上说，图像就是数据，当我们使用 imread 时，我们会将这些数据转换为我们非常熟悉的数据类型——NumPy 数组：

```python
# 查看图像类型
print(type(image))
```

![image-20220719095417519](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719095417519.png)

- 我们已将图像转换为一个矩阵，其元素对应于各个像素。 我们甚至可以看一下矩阵的实际值：

  ```python
  print(image)
  ```

  

![image-20220719095508114](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719095508114.png)

- 可以查看图像矩阵的大小

```python
print(image.shape)
```

![image-20220719095923870](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719095923870.png)

- 加载有色彩的图片

  - 初始时会加载成BGR格式(blue,green,red)

    ```python
    # 加载有颜色的图像
    image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
    # 展示像素
    print(image_bgr[0, 0])
    ```

  - 可以转换成RGB形式

  - 可以绘制

    ```python
    # 转换成RGB格式
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # 展示图片
    plt.imshow(image_rgb), plt.axis("off")
    plt.show()
    ```

    ![image-20220719100341862](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719100341862.png)

![image-20220719100353076](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719100353076.png)

- [(95条消息) 什么是RGB模式与BGR模式_SGchi的博客-CSDN博客_rgb和bgr](https://blog.csdn.net/sgchi/article/details/104474976)

### 8.2 Saving Images

- 保存图片

- `opencv's imwirite`

  saving.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 加载 image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 保存
print(cv2.imwrite("images/plane_new.jpg", image))
```

![image-20220719100605346](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719100605346.png)

![image-20220719100616602](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719100616602.png)

保存成功



### Discussion

- 图象格式根据文件名后缀确定（.jpg,.png etc.)
- imwrite 将覆盖现有文件而不输出错误或要求确认。



### 8.3 Resizing Images

- 更改图片大小
- `resize`

```python
# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 更改大小
image_50x50 = cv2.resize(image, (50, 50))
# 查看新的图片
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()
```

![image-20220719101110462](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719101110462.png)

#### Discussion

- 调整图像大小是图像处理常见的任务
- 图像有各种形状和大小，要用作特征，图像必须具有相同的尺寸
- 机器学习可能需要数千或数十万张图像。 当这些图像非常大时，它们会占用大量内存，通过调整它们的大小，我们可以显着减少内存使用量。

- 机器学习的一些常见图像尺寸是 32 × 32、64 × 64、96 × 96 和 256 × 256。





### 8.4 Cropping Images

- 对图像进行裁剪

- 切片

  croppingImage.py

```python
# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image in grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Select first half of the columns and all rows
image_cropped = image[:,:128]
# Show image
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()
```

![image-20220719102118362](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719102118362.png)

#### Discussion

- 裁剪对于只研究感兴趣的部分很有用



### 8.5 Blurring Images

- 使图像像素变得平滑，这样就可以模糊图像
- 求相邻像素的平均值

blurringImage.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 模糊
image_blurry = cv2.blur(image, (5,5))
# 展示
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

# 以100*100为区域均值模糊图像
image_very_blurry = cv2.blur(image, (100,100))
# Show image
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20220719103037650](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719103037650.png)卷积核大小5*5求均值模糊图像

![image-20220719103135583](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719103135583.png)卷积核大小为100*100求均值模糊图像

#### Discussion

- 什么是卷积核？

​	This neighbor and the operation performed are mathematically represented as a kernel (don’t worry if you don’t know what a kernel is).

​	相邻像素以及对其在数学上的操作叫做kernel（中文翻译为卷积核）

[(95条消息) 图像处理中的卷积核kernel_coder_by的博客-CSDN博客_卷积核](https://blog.csdn.net/i_silence/article/details/116483732)

```python
#我们在案例中使用的卷积核如下
# 创造 kernel
kernel = np.ones((5,5)) / 25.0
# 展示 kernel
print(kernel)
```

![image-20220719103837572](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719103837572.png)

- 中心元素是被检验的元素，而它周围的元素是邻居。因为值都是一样的，所以每一个元素对结果都有相同的权重。

- 我们可以使用`filter2D`手动实现卷积核应用于图像来达到类似的效果

  ```python
  # 运用卷积核
  image_kernel = cv2.filter2D(image, -1, kernel)
  # Show image
  plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
  plt.show()
  ```

  ![image-20220719114018584](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719114018584.png)

  [(95条消息) Python-OpenCV中的filter2D()函数_Mr.Idleman的博客-CSDN博客](https://blog.csdn.net/qq_42059060/article/details/107660265?ops_request_misc=%7B%22request%5Fid%22%3A%22165820238616780366584409%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=165820238616780366584409&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-107660265-null-null.142^v32^pc_rank_34,185^v2^control&utm_term=filter2d函数代码 python&spm=1018.2226.3001.4187)



### 8.6 Sharpening Images

- 锐化图像

- `filter2D`

  sharpenImage.py

  ```python
  # Load libraries
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  
  # Load image as grayscale
  image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
  # 创建卷积核
  kernel = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])
  # 锐化
  image_sharp = cv2.filter2D(image, -1, kernel)
  # 显示图片
  plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
  plt.show()
  
  ```

  

![image-20220719114544964](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719114544964.png)

#### Discussion

- 什么是图像锐化？

  图像锐化与图像平滑是相反的操作，锐化是通过增强高频分量来减少图像中的模糊，增强图像细节边缘和轮廓，增强灰度反差，便于后期对目标的识别和处理。锐化处理在增强图像边缘的同时也增加了图像的噪声。方法通常有**微分法**和**高通滤波法**。

- [(95条消息) 图像增强—图像锐化_白水baishui的博客-CSDN博客_图像锐化](https://blog.csdn.net/baishuiniyaonulia/article/details/98480583?ops_request_misc=%7B%22request%5Fid%22%3A%22165820254216782388074739%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=165820254216782388074739&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~hot_rank-3-98480583-null-null.185^v2^control&utm_term=锐化图像&spm=1018.2226.3001.4450)

- 本例子用的是高通滤波的一种算子

  ```python
  [[0, -1, 0],
   [-1, 5, -1],
   [0, -1, 0]]
  ```

  

### 8.7 Enhancing Contrast

- 增强图像之间像素的对比度

- 直方图均衡化是一种图像处理工具

- 我们有灰度图像时，我们可以直接在图像上应用 OpenCV 的 `equalizeHist`：

  enhanceContrast.py

  ```python
  # Load libraries
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  # Load image
  image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
  # 增强图像
  image_enhanced = cv2.equalizeHist(image)
  # 显示
  plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
  plt.show()
  
  
  # 有色图像
  image_bgr = cv2.imread("images/plane.jpg")
  # 转换成  YUV 形式
  image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
  # 直方图均衡化
  image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
  # 转换成RGB
  image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
  # 展示图像
  plt.imshow(image_rgb), plt.axis("off")
  plt.show()
  
  ```

  

![image-20220719135115090](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719135115090.png)

![image-20220719135644640](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719135644640.png)

#### Discussion

- 在处理有色图像时，需要先将图像转换成YUV的格式：Y 是亮度或亮度，U 和 V 表示颜色。 转换后，我们可以将 equalizeHist 应用于图像，然后将其转换回 BGR 或 RGB： 

- 直方图均衡如何工作的详细解释超出了本书的范围，但简短的解释是它会转换图像，以便使用更广泛的像素强度。

- 虽然生成的图像通常看起来并不“真实”，但我们需要记住，图像只是底层数据的视觉表示。 如果直方图均衡能够使感兴趣的对象更容易与其他对象或背景区分开来（并非总是如此），那么它可以成为我们图像预处理管道的有价值的补充

#### [直方图均衡化 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/44918476)

​	学习了一下，直方图均衡化的核心问题是推导出映射函数f和CDF

​	最简单的处理就是把理想中的函数想成均匀的，CDF认为是256

![image-20220719141431574](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719141431574.png)

当然有许多复杂的形式，可以让f更加符合局部的图片特征，在文章里有介绍





### 8.8 Isolating Colors

- 分离出图像颜色
- Define a range of colors and then apply a mask to the image

isolateColor.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image
image_bgr = cv2.imread('images/plane.jpg')
# 转化 BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# 定义两种蓝色 in HSV被隔离
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])
# 生成蒙板
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# 进行蒙板
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
# 转换 BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# 显示 image
plt.imshow(image_rgb), plt.axis("off")
plt.show()
```

![image-20220719142335088](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719142335088.png)

#### Discussion

- HSV:色调、饱和度和值
- mask在图像领域被翻译成蒙板
- 定义了一系列我们想要隔离的值，这可能是最困难和最耗时的部分。（在案例中是两种blue）
- 最后生成蒙板，蒙板也是二进制表示，可以进行按位与bitwise_and(src1, src2, dst=None, mask=None)





### 8.9 Binarizing Images

- 将图像黑白化（2值化）
- `Thresholding`,` adaptive thresholding`



threholding.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_grey = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 运用 adaptive thresholding
# 设置极值和邻居大小、均值操作值
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)
# 显示
plt.imshow(image_binarized, cmap="gray"), plt.axis("off")
plt.show()

```

![image-20220719145150260](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719145150260.png)

#### Discussion

- max_output_value 只是确定输出像素强度的最大强度。 

- cv2.ADAPTIVE_THRESH_GAUSSIAN_C 将像素的阈值设置为相邻像素强度的加权和。

- 权重由Gaussian window.

- 我们可以使用 cv2.ADAPTIVE_THRESH_MEAN_C 将阈值简单地设置为相邻像素的平均值

  ```python
  # Apply cv2.ADAPTIVE_THRESH_MEAN_C
  image_mean_threshold = cv2.adaptiveThreshold(image_grey,
                                               max_output_value,
                                               cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY,
                                               neighborhood_size,
                                               subtract_from_mean)
  # 展示
  plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off")
  plt.show()
  
  ```

   ![image-20220719145357949](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719145357949.png)

- 最后两个参数是块大小（用于确定像素阈值的邻域大小）和从计算的阈值中减去的常数（用于手动微调阈值）。
- 阈值化的一个主要好处是对图像进行去噪——只保留最重要的元素。



### 8.10 Removing Backgrounds

- 除去背景
- `GrabCut`算法

```python
# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to RGB
image_bgr = cv2.imread('images/background.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)
# 创建起始遮罩
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# 为grabCut算法使用的临时空间
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# 应用 grabCut
cv2.grabCut(image_rgb,  # 原图片
            mask,  # 初始遮罩
            rectangle,  # 定义的长方形区域
            bgdModel,  # 背景
            fgdModel,  # 背景
            5,  # Number of iterations
            cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle
# 将确定为背景的地方标记为0，否则标记为1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 把mask2减去
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# 显示
plt.imshow(image_rgb_nobg), plt.axis("off")

plt.show()

```

![image-20220719151808470](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719151808470.png)

![image-20220719151829945](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719151829945.png)

效果有点……差

#### Disscussion

- 首先作者承认`GrabCut`无法去除所有背景

- 在我们的解决方案中，我们首先在包含前景的区域周围标记一个矩形。 GrabCut 假设这个矩形之外的所有东西都是背景，并使用该信息来确定正方形内可能是什么背景（要了解算法如何做到这一点，请查看此解决方案末尾的外部资源）。 然后创建一个掩码，表示不同的确定/可能的背景/前景区域。

  ```python
  # Show mask
  plt.imshow(mask_2, cmap='gray'), plt.axis("off")
  plt.show()
  ```

  

![image-20220719152211553](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719152211553.png)

- 黑色区域是我们的矩形之外的区域，它被假定为绝对背景。灰色区域是 GrabCut 认为可能的背景，而白色区域可能是前景。 然后使用此蒙版创建合并黑色和灰色区域的第二个蒙版： 

  ```python
  # Show mask
  plt.imshow(mask_2, cmap='gray'), plt.axis("off")
  plt.show()
  ```

  

![image-20220719152707953](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719152707953.png)

#### 包含关于Grab Cut算法的介绍[(95条消息) 图像分割经典算法--《图割》（Graph Cut、Grab Cut-----python实现）_我的她像朵花的博客-CSDN博客_图割算法](https://blog.csdn.net/mmm_jsw/article/details/83866624)

### 8.11 Detecting Edges

- 查找图片中的边界

- `Canny edge detector`

detectEdges.py

```python
# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_gray = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 计算中值强度
median_intensity = np.median(image_gray)
# 设置阈值
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# 运用 canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
# 显示
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()

```

![image-20220719153718399](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719153718399.png)

#### Discussion

- 边缘检测是计算机视觉中的一个主要话题。边缘很重要，因为它们是高信息区域。
- 有许多边缘检测技术（Sobel 滤波器、拉普拉斯边缘检测器等）。但是，我们的解决方案使用常用的 Canny 边缘检测器。
- Canny 检测器需要两个参数来表示低梯度阈值和高梯度阈值。低阈值和高阈值之间的潜在边缘像素被认为是弱边缘像素，而高于高阈值的那些被认为是强边缘像素。

#### [(95条消息) Canny边缘检测_saltriver的博客-CSDN博客_canny](https://blog.csdn.net/saltriver/article/details/80545571)



### 8.12 Detecting Corners

- 检测出图像中的corner
- `cornerHarris`

detectCorners.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_bgr = cv2.imread("images/plane.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)
# 设置 corner detector 参数
block_size = 2
aperture = 29
free_parameter = 0.04
# 搜索corner
detector_responses = cv2.cornerHarris(image_gray,  # 原图
                                      block_size,  # 每个像素周围的邻居大小
                                      aperture,  # 使用的Sobel核大小
                                      free_parameter)  # 自由参数，越大可以识别越软的corner
# 将探测后的结果存储
detector_responses = cv2.dilate(detector_responses, None)
# 只要探测到的值大于阈值（这里是0.02的比例），设置成黑色
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255, 255, 255]
# 转换成灰度图像
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# 显示
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

```

![image-20220719155445106](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719155445106.png)

#### Discussion

- Harris角点检测器是检测两条边相交的常用方法。

- 我们对检测角点的兴趣与删除边缘的原因相同：角点是高信息点。哈里斯角检测器的完整解释可以在本节末尾的外部资源中找到，但一个简化的解释是它会寻找窗口（也称为邻域或补丁），其中窗口有小的移动（想象摇动窗口）在窗口内的像素内容中产生很大的变化。

- cornerHarris 包含三个重要的参数，我们可以使用它们来控制检测到的边缘。首先，block_size 是用于角点检测的每个像素周围的邻居的大小。其次，孔径是使用的 Sobel 核的大小（如果您不知道那是什么，请不要担心），最后还有一个自由参数，其中较大的值对应于识别较软的角。

  ```python
  # 显示可能的 corners
  plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
  plt.show()
  ```

![image-20220719155707152](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719155707152.png)

- 然后，我们应用阈值处理以仅保留最可能的角点。或者，我们可以使用类似的检测器 Shi-Tomasi 角检测器，它的工作方式与 Harris 检测器 (goodFeaturesToTrack) 类似，可以识别固定数量的强角。 

  ```python
  #  使用goodFeaturesToTrack
  # Load images
  image_bgr = cv2.imread('images/plane.jpg')
  image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  # Number of corners to detect
  corners_to_detect = 10
  minimum_quality_score = 0.05
  minimum_distance = 25
  # 检测
  corners = cv2.goodFeaturesToTrack(image_gray,
                                    corners_to_detect,  # 角点个数
                                    minimum_quality_score,  # 最低阈值
                                    minimum_distance)  # 最短的距离
  corners = np.float32(corners)
  # 圈出每个角点
  for corner in corners:
      x, y = corner[0]
      cv2.circle(image_bgr, (int(x), int(y)), 10, (255, 255, 255))
  # Convert to grayscale
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  # Show image
  plt.imshow(image_rgb, cmap='gray'), plt.axis("off")
  plt.show()
  
  ```

  ![image-20220719160600483](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719160600483.png)



### 8.13 Creating Features for Machine Learning

- 创建可以用于机器学习的特征
- `flatten`

features.py

```python
# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))
# Convert image data to one-dimensional vector
print(image_10x10.flatten())
```

![image-20220719161228352](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719161228352.png)

#### Discussion

- 如果是灰度图像一个像素一个value

  ```python
  plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
  plt.show()
  ```

  ![image-20220719161328275](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719161328275.png)

- 如果图像是彩色的，则不是每个像素都由一个值表示，而是由多个值（通常是三个）表示，这些值表示混合以形成最终颜色的通道（红色、绿色、蓝色等） 像素。

```python
# Load image in color
image_color = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))
# Convert image data to one-dimensional vector, show dimensions
print(image_color_10x10.flatten().shape)
```

![image-20220719161538190](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719161538190.png)

- 计算机视觉的一大挑战就是如何处理彩色图片增大，因为它每个像素都是一组特征，随之而来的特征数激增的问题

```python
# Load image in grayscale
image_256x256_gray = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_gray.flatten().shape)

# Load image in color
image_256x256_color = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_color.flatten().shape)
```

![image-20220719161901652](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719161901652.png)

如输出所示，即使是一张小的彩色图像也有近 200,000 个特征，这在我们训练模型时可能会出现问题，因为特征的数量可能远远超过观察的数量。这个问题将激发后面章节中讨论的维度策略，它试图减少特征的数量，同时不丢失数据中包含的过多信息。





### 8.14 Encoding Mean Color as a Feature

- 求出平均颜色

  meancolor.py

  ```python
  # Load libraries
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  # 加载
  image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
  # 计算每个channel的平均值
  channels = cv2.mean(image_bgr)
  # 交换blue和red的值 (making it RGB, not BGR)
  observation = np.array([(channels[2], channels[1], channels[0])])
  # 展示 mean channel values
  print(observation)
  
  # 显示
  plt.imshow(observation), plt.axis("off")
  plt.show()
  
  ```

  ![image-20220719162303067](C:\Users\12587\AppData\Roaming\Typora\typora-user-images\image-20220719162303067.png)

![image-20220719162255927](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719162255927.png)

#### Discussion

这三个颜色是每个channel的平均值，这可以作为图片的一个特征





### 8.15 Encoding Color Histograms as Features

- 生成一组代表颜色的特征值

- 计算每一种颜色的直方图

histograms.py

```python
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# 转换成 RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 特征
features = []
# 计算每一个channel
colors = ("r", "g", "b")
# 生成直方图
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 原图
                             [i],  # 索引
                             None,  # 遮罩
                             [256],  # 直方图大小
                             [0, 256])  # 范围
    features.extend(histogram)
# 创建一个用于表示特征的向量
observation = np.array(features).flatten()
# 展示前五项
print(observation[0:5])
```

![image-20220719162937324](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719162937324.png)



#### Discussion

- RGB每个有三个通道

  ```python
  # Show RGB channel values
  print(image_rgb[0,0])
  ```

  ![image-20220719163115097](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719163115097.png)

- 可以绘制直方图（pandas)

  ```python
  # 绘制直方图
  # Import pandas
  import pandas as pd
  # Create some data
  data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])
  # 显示
  data.hist(grid=False)
  plt.show()
  
  ```

  ![image-20220719163242106](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719163242106.png)

```python
# 计算每一个channel
colors = ("r", "g", "b")
# 生成直方图
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 原图
                             [i],  # 索引
                             None,  # 遮罩
                             [256],  # 直方图大小
                             [0, 256])  # 范围
    features.extend(histogram)
# 绘制
plt.plot(histogram, color=channel)
plt.xlim([0, 256])
# Show plot
plt.show()

```

![image-20220719163555161](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220719163555161.png)





## Chapter 9. Dimensionality Reduction Using Feature Extraction

### 9.0 Introduction

- 访问数十万的feature是很常见的。
- 幸运的是，并非所有特征都是平等的，降维特征提取的目标是转换我们的特征集 poriginal ，以便我们最终得到一个新的集合 pnew ，其中 poriginal > pnew ，同时仍然保留大部分基础信息。换句话说，我们减少了特征的数量，而我们的数据生成高质量的能力只有很小的损失 预测。
- 在本章中，我们将介绍一些特征提取技术来做到这一点。 我们讨论的特征提取技术的一个缺点是我们生成的新特征将无法被人类解释。它们将包含与训练我们的模型一样多或几乎一样多的能力，但在人眼中将显示为随机数的集合。如果我们想保持解释模型的能力，通过特征选择降维是更好的选择。



### 9.1 Reducing Features Using Principal Components

- 使特征降维，但是保持数据中的方差
- scikit’s `PCA`

PCA.py

```python
# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
# 加载数据
digits = datasets.load_digits()
# 是数据集标准化
features = StandardScaler().fit_transform(digits.data)
# 创建一个保留99%方差的PCA
pca = PCA(n_components=0.99, whiten=True)
# 生成一个PCA features
features_pca = pca.fit_transform(features)
# 显示
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])
```

![image-20220720101801823](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720101801823.png)

#### Discussion

- Principal component analysis (PCA) 是一种流行的降维技术
- PCA 将观察结果投射到特征矩阵中保留最大方差的（希望更少）主成分上。
- PCA 是一种无监督技术，这意味着它不使用来自目标向量的信息，而只考虑特征矩阵。
- PCA 是在 scikit-learn 中使用 pca 方法实现的。
  - n_components 有两个操作，具体取决于提供的参数。如果n_components大于 1，它将返回参数所指定数量的特征，这导致了如何选择最优特征数量的问题。对我们来说，如果 n_components 的参数介于 0 和 1 之间，pca 会返回保留方差的最小特征量。我们通常使用 0.95 和 0.99 的值，这意味着分别保留了原始特征的 95% 和 99% 的方差。
  - whiten=True 转换每个主成分的值，使它们具有零均值和单位方差。（标准化）
  - 还有一个参数是svd_solver="randomized"，它实现了一种随机算法，通常会以更短的时间找到第一个主成分。

#### PCA的数学原理

- 原书并没有介绍非常详细，只是简单举了一个例子

- 在这里看了几篇资料比较详细的是[【机器学习】降维——PCA（非常详细） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/77151308)

- 概述：PCA（Principal Component Analysis） 是一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量。

- 原理：

  - 首先我们在数学上学过基的概念，如果想要把一个N维向量投影到k维，需要选择K个基，那么如何选择K个基使这N个向量保留原有的信息就是我们需要解决的问题；
  - 我们也学过方差（协方差），方差在一维上表示数值的分散程度，那么我们上述的问题其实也就简化成了所有数据变换转换到对应的基上，数据在这个基的方差最大
  - 为了方便计算我们需要对数据进行标准化（也就对应了案例中执行`features = StandardScaler().fit_transform(digits.data)`的语句）
  - ![image-20220720104339749](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720104339749.png)

  - 那么最后我们的问题就化简成了：**将一组 N 维向量降为 K 维，其目标是选择 K 个单位正交基，使得原始数据变换到这组基上后，各变量两两间协方差为 0，而变量方差则尽可能大（在正交的约束下，取最大的 K 个方差）。**
  - 根据我们的优化条件，**我们需要将除对角线外的其它元素化为 0，并且在对角线上将元素按大小从上到下排列（变量方差尽可能大）**
  - **矩阵对角化**：原数据协方差矩阵C和转换后的协方差矩阵D满足：
  - ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++D+%26+%3D++%5Cfrac%7B1%7D%7Bm%7DYY%5ET+%5C%5C++%26+%3D+%5Cfrac%7B1%7D%7Bm%7D%28PX%29%28PX%29%5ET+%5C%5C+%26+%3D+%5Cfrac%7B1%7D%7Bm%7DPXX%5ETP%5ET+%5C%5C++%26+%3D+P%28%5Cfrac%7B1%7D%7Bm%7DXX%5ET%29P%5ET+%5C%5C++%26+%3D+PCP%5ET++%5Cend%7Baligned%7D++%5C%5C)

  （P是一组基，X是原来的矩阵，Y是转换后的矩阵）

  - 协方差矩阵有这样的特性：

    - 是一个实对称矩阵：实对称矩阵不同特征值对应的特征向量必然正交。
    - 设特征向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 重数为 r，则必然存在 r 个线性无关的特征向量对应于 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) ，因此可以将这 r 个特征向量单位正交化。

    所以可以得出结论：协方差矩阵一定可以找到N个线性无关的单位正交向量，我们可以按列组成矩阵

    $E=(e_1,e_2,.....,e_N)$

  - 所以根据前面的优化条件我们已经可以得出我们所需的矩阵P，P是协方差矩阵特征向量单位化后按照从大到小的顺序排列出来的矩阵，其中每一行都是原矩阵C的特征向量，根据需要的维数将P压缩成P‘（前K行）$Y=P'X$得到降维后的Y

  - 总结就是6步：

    - 将原始数据按列组成 n 行 m 列矩阵 X；
    - 将 X 的每一行进行零均值化，即减去这一行的均值；（标准化）
    - 求出协方差矩阵 ![[公式]](https://www.zhihu.com/equation?tex=C%3D%5Cfrac%7B1%7D%7Bm%7DXX%5E%5Cmathsf%7BT%7D) ；
    - 求出协方差矩阵的特征值及对应的特征向量；
    - 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P；
    - ![[公式]](https://www.zhihu.com/equation?tex=Y%3DPX) 即为降维到 k 维后的数据。



### 9.2 Reducing Features When Data Is Linearly Inseparable（线性不可分）

- 线性可分就是说可以用一个线性函数把两类样本分开
- 目标是把线性可分的数据进行降维
- 使用拓展的PCA算法——Kernel PCA 对非线性数据进行降维

 linearlyInseparable.py

```python
# Load libraries
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
# 创建一个线性可分的数据 圆数据集
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# 使用 kernal PCA  核函数RBF（高斯核函数） 系数15，降维到1维
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])
```

![image-20220720111903218](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720111903218.png)

#### Discussion

- PCA用于降维。标准PCA用于线性情况的降维，如果是线性可分的数据集（可以用直线或者超平面将类分开）那么PCA的降维效果很好，反之，PCA降维效果就不行

- `make_circles`产生同心圆的数据集，这两个圆是线性可分的

  ![image-20220720112625658](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720112625658.png)

- 但在本问题中，如果我们像上一节一样使用线性的PCA那么因为两个圆是非线性的，他们的投影会交织在一起，效果很差

![image-20220720112614121](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720112614121.png)

- 理想情况下，我们需要一个既可以减少维度又可以使数据线性可分的转换。而KernelPCA就可以同时做到两个目标
- 内核允许我们将线性不可分的数据投影到线性可分的更高维度； **这称为kernel trick**
- Kernel（这里应该指的是核函数）允许我们将线性不可分的数据投影到线性可分的更高维度。
- KernelPCA可以允许提供很多内核包括poly(多项式)，gbf(高斯核)，sigmoid（sigmoid函数），甚至linear(线性函数）

- linear情况下的效果与线性PCA一模一样
- KernelPCA一个缺点就是需要我们指定很多参数`n_components`表示参数数量，核函数本身就需要参数，例如本例中gbf核就需要gamma值
- 我们将在12章讨论如何确定这些参数



#### KernelPCA的原理

[数据降维: 核主成分分析(Kernel PCA)原理解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/59775730)

- 详细的原理就不复述了，毕竟确实讲的很好
- 降维的大部分与之前PCA的一致，大多数不同点集中在转换到高维空间，以及Kernel的部分
- 需要注意的点有：
  - 非线性映射 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 将 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D) 中的向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D) 映射到高维空间(记为 ![[公式]](https://www.zhihu.com/equation?tex=D) 维)，但是这个![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)是不会显示指定的，只需要定义出高维特征空间的空间向量即可，这就是kernel trick的本质
  - 仍然一样，映射以后为了方便计算仍然需要进行中心化处理





### 9.3 Reducing Features by Maximizing Class Separability

- 通过分类器来减少特征
- `linear discriminant analysis (LDA)`

```python
# Load libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 莺尾花数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建lda
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# 打印
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])
# 方差
print(lda.explained_variance_ratio_)
```

![image-20220720142224884](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720142224884.png)

![image-20220720142425287](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720142425287.png)

#### Discussion

- LDA 是一种分类方法，也是一种流行的降维技术。
- 对比LDA的不同：
  - LDA算法的思想是将数据投影到低维空间之后，使得同一类数据尽可能的紧凑，不同类的数据尽可能分散。
  - LDA是有监督的机器学习算法
  - LDA根据的是均值，PCA根据的是方差
- 在 scikit-learn 中，LDA 是使用 LinearDiscriminantAnalysis 实现的， 其中包括一个参数 n_components，表示我们想要返回的特征数量。
- 具体来说，我们可以运行 LinearDiscriminantAnalysis 并将 n_components 设置为 None 以返回每个组件特征解释的方差比率，然后计算需要多少组件才能超过某个解释的方差阈值（通常为 0.95 或 0.99）

​	 官方文档表明：If None, will be set to min(n_classes - 1, n_features)

```python
# 测试参数
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)
# 方差值
lda_var_ratios = lda.explained_variance_ratio_
print(lda_var_ratios)


# 计算n_components多大时才能够达到goal_var的阈值
def select_n_components(var_ratio, goal_var: float) -> int:
    # 设置初始的参数
    total_variance = 0.0
    # 初始的特征数
    n_components = 0
    # 对于每个ratio计算
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


# 运行函数
print(select_n_components(lda_var_ratios, 0.99))
```

![image-20220720145713569](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720145713569.png)

#### LDA原理

[机器学习-LDA(线性判别降维算法) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/51769969)

[机器学习-降维方法-有监督学习：LDA算法(线性判别分析)【流程：①类内散度矩阵Sw-＞②类间散度矩阵Sb-＞计算Sw^-1Sb的特征值、特征向量W-＞得到投影矩阵W-＞将样本X通过W投影进行降维】_u013250861的博客-CSDN博客_有监督lda](https://blog.csdn.net/u013250861/article/details/121042287)

1. 首先明确目的：我们需要在降维后保持特征不重合的同时还要保证类内数据紧凑，类之间数据分散；

2. 我们需要把数据投影到一个超平面上（两类的就是直线上），那么现在就需要考虑如何确定这个超平面来保证类间和类内的点的距离

3. 假设就是2个类，投影在一条直线上，那么我们必须要保证类之间的距离最大，那么可以通过类中心来计算：假设$\mu_1是第一个类的中心，\mu_2是第二个类的中心$，那么距离计算就是

   $|w^T\mu_1-w^T\mu2|$($w^T$是投影方向转置)

4. 同时要类内距离最短，我们就需要类内的差距最小，那么自然而然就想到的是方差的计算方式，不过在这里定义的不是协方差矩阵而是散度矩阵（链接2其实介绍了两者本质无区别只是差了个常数）

​	$\Sigma_j(j=1,2) = \Sigma_{x\in X_j}(x-\mu_j)(x-\mu_j)^T$

​	那么需要最小化$w^T\Sigma_0w+w^T\Sigma_1w$

5、我们定义类内散度矩阵为：$S_w=\Sigma_0+\Sigma_1$,类间散度矩阵为$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$

6、最后我们得到目标$arg\ max\ J(w)=w^TS_bw/w^TS_ww$

7、求J对W的偏导数取0，此时J最大，化简等式得到：$(w^TS_Bw)S_ww=(w^TS_ww)S_Bw$

8、由于（7）式内用括号括起来的都只是一个值，我们定义$\lambda=w^TS_Bw/w^TS_ww$

9、根据（7）（8）化简得到$S_w^{-1}S_Bw=\lambda w$最后只需要按照特征向量的求法，求出$w$即可得到我们需要的投影方向



### 9.4 Reducing Features Using Matrix

- 对一个没有负值的矩阵进行降维
- NMF算法

NMF.py

```python
# Load libraries
from sklearn.decomposition import NMF
from sklearn import datasets
# 加载data
digits = datasets.load_digits()
# 加载特征矩阵
features = digits.data
# 创建、转换和应用 NMF
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)
# 展示结郭
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])
```

![image-20220720154228980](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720154228980.png)

#### Discussion

- non-negative matrix factorization (NMF) 是一种用于线性降维的无监督技术，它将特征矩阵分解（即分解为乘积近似于原始矩阵的多个矩阵）为表示观察值与其特征之间潜在关系的矩阵。
- $V=WH$   形式上，给定需要返回的特征 r，NMF 将我们的特征矩阵分解为： 其中 V 是我们的 d × n 特征矩阵（即 d 个特征，n 个观察值），W 是一个 d × r，H 是一个 r × n 矩阵。通过调整 r 的值，我们可以设置所需的降维量。
- NMF 没有为我们提供输出特征的解释方差。因此，我们找到 n_components 最佳值的最佳方法是尝试一系列值，以找到在最终模型中产生最佳结果的值（参见第 12 章）。

#### NMF原理

- 实际上上网搜索发现原书已经将基本的NMF原理讲出来了，所以没有什么其他的
- 但是对于更详细的信息网上可以参考这篇博客：[NMF 非负矩阵分解 -- 原理与应用_qq_26225295的博客-CSDN博客_nmf原理](https://blog.csdn.net/qq_26225295/article/details/51211529)
- 其实大部分NMF的问题不是原理问题，而是K值，也就是这个n_components的参数怎么定才合适



### 9.5 Reducing Features on Sparse Data

- 对稀疏矩阵降维
- `TSVD`

TSVD.py

```python
# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
# Load the data
digits = datasets.load_digits()
# 标准化
features = StandardScaler().fit_transform(digits.data)
# 稀疏化
features_sparse = csr_matrix(features)
# 创建一个 TSVD
tsvd = TruncatedSVD(n_components=10)
# 应用 TSVD
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
# 显示
print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])
```

![image-20220720160027171](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720160027171.png)

#### Discussion

- Truncated Singular Value Decomposition(TSVD) 截断奇异值分解

- TSVD 类似于 PCA，事实上，PCA 实际上经常在其步骤之一中使用非截断奇异值分解 (SVD)。TSVD 的实际优势在于，与 PCA 不同，它适用于稀疏特征矩阵。

- TSVD 的一个问题是，由于它使用随机数生成器的方式，输出的符号可以在拟合之间翻转。一个简单的解决方法是每个预处理管道只使用一次 fit，然后多次使用 transform。

- 与LDA一样，我们必须指定我们想要输出的特征（components）的数量。这是通过 n_components 参数完成的。

- 那么一个自然而然的问题就是：components的最佳数量是多少？

  - 一种策略是将 n_components 作为hyperparameter包含在模型选择期间进行优化（即，选择 n_components 的值以产生最佳训练模型）
  - 由于 TSVD 为我们提供了每个分量解释的原始特征矩阵方差的比率，我们可以选择解释所需方差量的分量的数量（95% 或 99% 是常用值）。

  ```python
  # 打印前3个维度上的方差和
  print(tsvd.explained_variance_ratio_[0:3].sum())
  ```

  ![image-20220720161820922](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720161820922.png)

```python
# 创建一个tsvd并运用
tsvd = TruncatedSVD(n_components=features_sparse.shape[1] - 1)
features_tsvd = tsvd.fit(features)
# 列出所有方差
tsvd_var_ratios = tsvd.explained_variance_ratio_


# 创建类似于第二节的function
def select_n_components(var_ratio, goal_var):
    total_variance = 0.0
    n_components = 0
    # 对于每一个比例的方差来说:
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        # 一旦方差大于指定的方差，返回
        if total_variance >= goal_var:
            break
    # Return the number of components
    return n_components


# 计算components
print(select_n_components(tsvd_var_ratios, 0.95))

```

![image-20220720161738865](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220720161738865.png)

#### TSVD原理

- [【zt】TSVD - 简书 (jianshu.com)](https://www.jianshu.com/p/8157d8fb0a74)
- [TSVD截断奇异值分解_像在吹的博客-CSDN博客_截断奇异值分解](https://blog.csdn.net/zhangweiguo_717/article/details/71778470)
- 网上的TSVD讲的比较少，因为TSVD是SVD的变形，所以大多都是讲SVD

- TruncatedSVD是SVD的变形，只计算用户指定的最大的K个奇异值（特征值）
- **TSVD实际上算法和PCA特别像**实际上就是把协方差矩阵变成矩阵本身





## Chapter 10. Dimensionality Reduction Using Feature Selection

### 10.0 Introduction

- 在第9章中我们介绍了如何创建（理想状态下）训练质量模型类似的但特征维度显著减少的新特征矩阵。**称为Feature Extraction**

- 在本章我们将如何介绍选择高质量、信息丰富的特征并删除不太有用的特征。这**称为Feature Selection（特征选择）**。
- 特征选择往往主要有三种方法：**filter, wrapper, and embedded.**
  - filter：检查特征的统计特性来选择最佳特征
  - wrapper： 使用试错法来找到产生具有最高质量预测的模型的特征子集。
  - embedded:选择最佳特征子集作为学习算法训练过程的一部分或扩展
  - embedded与特定的学习算法密切相关，在深入研究算法本身之前很难解释它们。
- 我们在本章中仅介绍过滤器和包装器特征选择方法，将特定嵌入式方法的讨论留到深入讨论这些学习算法的章节。



### 10.1 Thresholding Numerical Feature Variance

- 移除方差较小的特征
- 通过设置阈值来控制最小方差达到筛选的作用

thresholdingVariance.py

```python
# Load libraries
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
# import some data to play with
iris = datasets.load_iris()
# 创建特征矩阵和目标向量
features = iris.data
target = iris.target
# 创建 thresholder
thresholder = VarianceThreshold(threshold=.5)
# 使用thresholder仅仅保留方差大于0.5的特征
features_high_variance = thresholder.fit_transform(features)
# 打印前三个observation
print(features_high_variance[0:3])
```

![image-20220721101159632](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721101159632.png)

#### Discussion

- VT（variance thresholding) 是特征选择的最基本的方式之一。使用该方法的原因是我们认为方差较小的数据可能不如方差较大的数据有用。

- $var(x) = \frac{1}{n}\Sigma_{i=1}^{n}(x_i-\mu)^2$

- 注意：

  1. 方差是不居中的。也就是说当单位不同时，VT将不再起到作用
  2. 阈值是一个hyperparameter。也就是需要我们人为的来制定的。在第12章，我们会学到如何判断这个方差阈值是不是一个好的阈值

- 我们可以通过`variances_`查看每个特征的方差：

  ```python
  # 查看方差
  print(thresholder.fit(features).variances_)
  ```

  ![image-20220721103120717](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721103120717.png)



### 10.2 Thresholding Binary Feature Variance

- 有一组只有分类信息的特征，希望能够删除那些方差较低的特征
- 伯努利分布的方差

binaryThresholdingVariance.py

```python
# Load library
from sklearn.feature_selection import VarianceThreshold
# 创建的矩阵满足:
# 特征 0: 80% class 0
# 特征 1: 80% class 1
# 特征 2: 60% class 0, 40% class 1
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]
# 创建thresholder，并根据伯努利分布的方差声明参数threshold
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
print(thresholder.fit_transform(features))
```

![image-20220721103931853](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721103931853.png)

#### Discussion

- 伯努利分布就是2项分布，这种分布的方差满足$var(p)=p(1-p)$(p为某一类的比例)



### 10.3 Handling Highly Correlated Features

- 部分特征高度相关的数据集

- 如果存在某些高度相关的特征，我们就可以把它去除

  correlatedFeatures.py

```python
# Load libraries
import pandas as pd
import numpy as np

# 创建一个特征矩阵，可以很明显的看出前两个特征高度线性相关
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])
# 特征矩阵 -> DataFrame
dataframe = pd.DataFrame(features)
# 创建一个线性相关度的矩阵
corr_matrix = dataframe.corr().abs()
# 观察相关程度
print(corr_matrix)
# 选取这个矩阵的上三角部分（因为对称）
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k=1).astype(bool))
# 寻找特征之间线性相关度大于0.95的特征
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# 删除这些特征
print(dataframe.drop(dataframe.columns[to_drop], axis=1).head(3))
```

![image-20220721105320847](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721105320847.png)

#### Discussion

-  如果两个特征高度相关，那么它们包含的信息非常相似，同时包含这两个特征可能是多余的。



### 10.4 Removing Irrelevant Features for Classification

- 移除分类问题中的无关变量
- ` chi-square (χ 2 ) statistic`

irrelevantFeatures.py

```python
# Load libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
# 加载莺尾花数据
iris = load_iris()
features = iris.data
target = iris.target
# 分类数据变为数字
features = features.astype(int)
# 通过chi2来检验分类是否和这些特征有关
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])


# 选择具有很高的f-value(ANOVA方法）
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 选择topn的特征
from sklearn.feature_selection import SelectPercentile

# 选择前75%
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
```

![image-20220721112150790](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721112150790.png)



#### Discussion

- chi2（$χ^2$ 统计）判断是否观测值的特征与目标值相互独立。

  - $O_i是样本在第i类的数量，E_i是如果特征与结果没有关系，样本在第i类的数量的期望值$
  - $χ^2 = \Sigma_{i=1}^n \frac{(O_i-E_i)^2}{E_i}$

  - $χ^2$统计告诉您观察到的计数与如果总体中没有任何关系时您预期的计数之间存在多少差异。
  - 如果目标独立于特征变量，那么它与我们的目的无关，因为它不包含我们可以用于分类的信息。另一方面，如果这两个特征高度依赖，它们可能对训练我们的模型非常有用。

- 使用 SelectKBest 选择具有最佳统计信息的特征。参数 k 决定了我们想要保留的特征的数量。

- $χ^2$统计量只能在两个分类向量之间计算。出于这个原因，用于特征选择的卡方要求目标向量和特征都是分类的。但是，**如果我们有一个数值特征，我们可以通过首先将定量特征转换为分类特征来使用卡方技术。最后，要使用我们的卡方方法，所有值都必须是非负的。**

- 如果我们有一个数值特征，我们可以使用 f_classif 来计算每个特征和目标向量的 ANOVA F 值统计量。

  - f-classif检查当我们按目标向量对数值特征进行分组时，每组的均值是否显着不同。例如，如果我们有一个二元目标向量、性别和一个定量特征、测试分数，则 f-classif将告诉我们男性的平均测试分数是否不同于女性的平均测试分数。



### 10.5 Recursively Eliminating Features

- 自动的去选择那些最值得保留的特征

#### Solution

- `RFECV`生成一个使用cross-validation (CV)的recursive feature elimination (RFE) 

- 反复的训练一个模型，每一次移除一个特征,直到模型训练效果很差，那么保留下来的特征就是我们最需要的特征



recursiveEliminating.py

```python
# 加载库
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# 丢弃警告
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")
# 生成有10000个样本，100个特征的线性回归的样本
features, target = make_regression(n_samples=10000,
                                   n_features=100,
                                   n_informative=2,
                                   random_state=1)
# 创建一个线性模型
ols = linear_model.LinearRegression()
# 递归的去除特征
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
print(rfecv.transform(features))
# 查看好的特征的数量
print(rfecv.n_features_)

# 查看哪些特征应该被保留
print(rfecv.support_)

# 我们可以查看特征的排行
print(rfecv.ranking_)
```

![image-20220721114521641](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721114521641.png)

#### Discussion

- 作者认为RFE技术是1-10章最为先进的技术（说明很重要）

- RFE 背后的想法是重复训练一个包含一些参数（也称为权重或系数）的模型，例如线性回归或支持向量机

- 第一次训练模型时，我们包含了所有特征。然后，我们找到具有最小参数的特征（请注意，这假设特征被重新缩放或标准化），这意味着它不太重要，并从特征集中删除该特征。

- 我们应该保留多少特征？我们可以（假设地）重复这个循环，直到我们只剩下一个特征。更好的方法需要我们包含一个称为交叉验证（CV）的新概念。

  大体思路：

  - 给定包含 (1) 我们要预测的目标和 (2) 特征矩阵的数据，
  - 我们将数据分成两组：训练集和测试集
  - 我们使用训练集训练我们的模型。
  - 假装不知道测试集的目标，并将我们的模型应用于测试集的特征，以预测测试集的值
  - 将我们的预测目标值与真实目标值进行比较以评估我们的模型。

- 事实证明：将我们的预测目标值与真实目标值进行比较以评估我们的模型。

- 在 scikit-learn 中，带有 CV 的 RFE 是使用 RFECV 实现的，并且包含许多重要参数。

  - `estimator`:决定了我们要训练的模型类型(linear regression)
  - `step`:设置在每个循环期间要丢弃的特征的数量或比例
  - ` scoring`:设置了我们在交叉验证期间用于评估模型的质量指标

- 查阅资料得到更详细的RFECV的信息

  [sklearn.feature_selection.RFECV-scikit-learn中文社区](https://scikit-learn.org.cn/view/745.html)

| **estimator**              | **object** 一种监督学习估计器，其`fit`方法通过`coef_` 属性或`feature_importances_`属性提供有关特征重要性的信息。 |
| -------------------------- | ------------------------------------------------------------ |
| **step**                   | **int or float, optional (default=1)** 如果大于或等于1，则`step`对应于每次迭代要删除的个特征个数。如果在（0.0，1.0）之内，则`step`对应于每次迭代要删除的特征的百分比（向下舍入）。请注意，为了达到`min_features_to_select`，最后一次迭代删除的特征可能少于`step`。 |
| **min_features_to_select** | **int, (default=1)** 最少要选择的特征数量。即使原始特征数量与`min_features_to_select`之间的差不能被`step`整除， 也会对这些特征数进行评分。  *0.20版中的新功能。* |
| **cv**                     | **int, cross-validation generator or an iterable, optional** 确定交叉验证拆分策略。可能的输入是： - 无，要使用默认的5倍交叉验证 - 整数，用于指定折数。 - [CV分配器](http://scikit-learn.org.cn/lists/91.html#类API和估算器类型) - 可迭代的产生（训练，测试）拆分为索引数组。  对于整数或无输入，如果`y`是二分类或多分类， 则使用[`sklearn.model_selection.StratifiedKFold`](https://scikit-learn.org.cn/view/645.html)。如果估计器是分类器，或者`y`既不是二分类也不是多分类， 则使用[`sklearn.model_selection.KFold`](https://scikit-learn.org.cn/view/636.html)。  有关可在此处使用的各种交叉验证策略，请参阅[用户指南](http://scikit-learn.org.cn/view/6.html)。 *在0.22版本中更改：*`cv`无的默认值从3更改为5。 |
| **scoring**                | **string, callable or None, optional, (default=None)** 字符串(参见模型评估文档)或具有`scorer(estimator, X, y)`签名的scorer可调用对象或函数。 |
| **verbose**                | **int, (default=0)** 控制输出的详细程度。                    |
| **n_jobs**                 | **int or None, optional (default=None)** 跨折时要并行运行的核心数。 除非在上下文中设置了[`joblib.parallel_backend`](https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend)参数，否则`None`表示1 。 `-1`表示使用所有处理器。有关更多详细信息，请参见[词汇表](http://scikit-learn.org.cn/lists/91.html#参数)。 |









## Chapter 11. Model Evaluation

### 11.0 Introduction

- 在本章中，我们将研究如何评估我们学习算法所创建模型的质量。
- 从根本上说，机器学习不是创建模型，而是创建有高质量的模型
- 所以如何评估一个模型的好坏就显得尤为重要



### 11.1 Cross-Validating Models

- 交叉验证模型
- 创建一个管道来预处理数据、训练模型，然后使用交叉验证对其进行评估：

CV.py

```python
# Load libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 加载手写数字数据集
digits = datasets.load_digits()
# 特征矩阵
features = digits.data
# 目标向量
target = digits.target
# 标准化
standardizer = StandardScaler()
# 逻辑回归
logit = LogisticRegression()
# 复合估计器
pipeline = make_pipeline(standardizer, logit)
# 创建KFold cv
kf = KFold(n_splits=10, shuffle=True, random_state=1)
# 执行 k-fold cross-validation
cv_results = cross_val_score(pipeline,  # Pipeline
                             features,  # Feature matrix
                             target,  # Target vector
                             cv=kf,  # Cross-validation technique
                             scoring="accuracy",  # Loss function
                             n_jobs=-1)  # Use all CPU scores
# 计算平均值
print(cv_results.mean())
```

![image-20220721145747493](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721145747493.png)

#### Discussion

- 我们的目标不是评估模型在我们的训练数据上的表现如何，而是它在从未见过的数据（例如，新客户、新犯罪、新图像）上的表现如何。出于这个原因，我们的评估方法应该帮助我们了解模型能够如何从他们从未见过的数据中做出预测。

- 一种策略可能是推迟一部分数据进行测试。这称为验证（或保留）。

  - 在验证中，我们的观察（特征和目标）分为两组，传统上称为训练集和测试集。我们把测试集放在一边，假装我们以前从未见过它。
  - 我们使用我们的训练集训练我们的模型，使用特征和目标向量来教模型如何做出最佳预测。
  - 我们通过评估我们在训练集上训练的模型在测试集上的表现来模拟从未见过的外部数据。
  - 验证法的缺点：
    - 模型的性能可能高度依赖于为测试集选择的少数观测值。
    - 模型没有使用所有可用数据进行训练，也没有根据所有可用数据进行评估。

- 更好的策略：KFCV —— k重交叉验证

  - 在 KFCV 中，我们将数据分成 k 个部分，称为“折叠”。 然后使用 k – 1 折对模型进行训练——合并为一个训练集——然后将最后一个折用作测试集.

  - 我们重复这 k 次，每次使用不同的折叠作为测试集。 然后对每个 k 次迭代的模型性能进行平均以产生整体测量。

  - 在案例中，我们使用了10折进行KFCV，并将结果输出到cv_results

    ```python
    # 查看结果
    print(cv_results)
    ```

    ![image-20220721150543506](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721150543506.png)

  - KFCV的三个要点：

    - IID:首先，KFCV 假设每个观察都是独立于另一个创建的（即数据是独立同分布). 如果数据是独立同分布的，那么在分配给折叠时打乱观察是个好主意。 在 `scikit-learn` 中，我们可以设置`shuffle=True` 来执行洗牌。

    - 通常在k个folder中，每个类所占的百分比基本相同。在` scikit-learn` 中，我们可以通过将 `KFold` 类替换为 `StratifiedKFold `来进行分层 k 折交叉验证。

    - 当我们使用验证集或交叉验证时，重要的是基于训练集预处理数据，然后将这些转换应用于训练集和测试集。如，当我们进行标准化时，我们只计算训练集的均值和方差（fit)。 然后我们将该`StandardScaler`(transform)到应用于训练集和测试集：

      ```python
      # Import library
      from sklearn.model_selection import train_test_split
      # 创建测试集和训练集
      features_train, features_test, target_train, target_test = train_test_split(
      features, target, test_size=0.1, random_state=1)
      # Fit 训练集，不运用到测试集
      standardizer.fit(features_train)
      # transform 训练集和测试集
      features_train_std = standardizer.transform(features_train)
      features_test_std = standardizer.transform(features_test)
      
      ```

      

      这样做的原因是因为我们假装测试集是未知数据。如果我们使用来自训练集和测试集的观察来拟合我们的预处理器，则来自测试集的一些信息会泄漏到我们的训练集中。此规则适用于任何预处理步骤，例如特征选择。

  - scikit-learn’s `pipeline` package:

    - `pipeline`包很容易做到上述第三点，首先我们创建一个pipeline；

    - 然后我们使用该管道运行 KFCV，scikit 为我们完成所有工作； 

      ```python
      # 创建 pipeline
      pipeline = make_pipeline(standardizer, logit)
      # Do k-fold cross-validation scikit 为我们完成所有工作
      cv_results = cross_val_score(pipeline, # 管道
      							features, # 特征矩阵
      							target, # 目标向量
      							cv=kf, # CV 采用的方法，这里是KFolder
      							scoring="accuracy", # 损失函数，这里是准确度
      							n_jobs=-1) # 使用所有CPU来评估，如果是正数并行的数量
      ```

    - cross_val_score 带有三个值得注意的参数

      - cv 决定了我们的交叉验证技术。常用的有：**KFold、LeaveOneOut**

        - scoring损失函数：常见的有分类型：‘precision’ （准确率）和 ’recall‘ （召回率）和 ’f1‘（前两者的调和平均数），聚类型：adjusted_mutual_info_score、回归型：neg_mean_squared_error

        文档：[3.3. Metrics and scoring: quantifying the quality of predictions — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

        - 并行评估的数量，-1表示用所有处理器

### 11.2 Creating a Baseline Regression Model

- 使用一个baseline（理解为朴素的、启发式的）的线性回归模型和自己的模型结果作比对
- `DummyRegressor`

dummyRegressorExample.py

```python
# Load libraries
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# 波士顿房价数据集被移除，换成fetch_california_housing
housing = fetch_california_housing()
# 特征矩阵、目标
features, target = housing.data, housing.target
# 分离测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)
# 创建一个DummyRegressor
dummy = DummyRegressor(strategy='mean')
# 训练它
dummy.fit(features_train, target_train)
# 得到R^2得分
print(dummy.score(features_test, target_test))

# Load library
from sklearn.linear_model import LinearRegression

# 训练一个简单的线性模型
ols = LinearRegression()
ols.fit(features_train, target_train)
# 得分
print(ols.score(features_test, target_test))
```

​		

![image-20220721154743527](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721154743527.png)

#### Discussion

- DummyRegressor 允许我们创建一个非常简单的模型，我们可以将其用作基线来与我们的实际模型进行比较。
- DummyRegressor 使用 strategy 参数设置进行预测的方法，包括训练集中的均值或中值。 此外，如果我们将策略设置为常量并使用常量参数，我们可以设置虚拟回归器来预测每个观察值的某个常量值：

```python
# 每一次都预测为20
clf = DummyRegressor(strategy='constant', constant=20)
clf.fit(features_train, target_train)
# 得分
print(clf.score(features_test, target_test))
```

![image-20220721155032595](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721155032595.png)

- $R^2$评分：

  $R^2 = 1- \frac{\Sigma_i(y_i-\hat{y_i})}{\Sigma_i(y_i-\overline{y})}$

  $\hat{y_i}$是预测值，$\overline{y}$是目标均值

  $R^2$越接近1，目标向量中由特征解释的方差越大，拟合越好。



### 11.3 Creating a Baseline Classification Model

- 创建一个baseline的分类器
- `DummyClassifier`

dummyClassifier.py

```python
# Load libraries
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# 莺尾花数据
iris = load_iris()
# 创建target vector  feature matrix
features, target = iris.data, iris.target
# 分离训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)
# 创建 dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)
# "训练" model
dummy.fit(features_train, target_train)
# 获得准确性的评分
print(dummy.score(features_test, target_test))


# Load library
from sklearn.ensemble import RandomForestClassifier
# 创建一个随机森林的分类器（14章会介绍原理）
classifier = RandomForestClassifier()
# 训练模型
classifier.fit(features_train, target_train)
# 得到准确性的评分
print(classifier.score(features_test, target_test))
```

![image-20220721222637016](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721222637016.png)

#### Discussion

- 分类器的一个常见的评判标准就是它比随机预测好多少。
- `DummyClassifier`就是进行随机猜测。
- 一个重要的参数是`strategy`:
  - stratified:返回类的概率就是训练集中类的频率；
  - uniform：从 y 中观察到的唯一类列表中随机均匀地生成预测，即每个类具有相等的概率。
  - prior：默认值。predict总是返回最频繁出现的类
  - 其他取值：[sklearn.dummy.DummyClassifier — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)



### 11.4 Evaluating Binary Classifier Predictions

- 给定一个已经训练好的二分类模型，评价它的质量

#### Solution:二分类器评价原理

- 使用 scikit-learn 的 cross_val_score 进行交叉验证，同时使用打分参数定义多个性能指标之一，包括准确度、精确度、召回率和 F1。
- 准确性是一种常见的性能指标。 它只是正确预测的观察值的比例：
  - $Accurancy = \frac{TP+TN}{TP+TN+FP+FN}$
  - TP:真阳性数量
  - TN：真阴性数量
  - FP:假阳性数量
  - FN：假阴性数量
  - 参数`scoring="accuracy"`
  - 缺点：在存在高度不平衡的类模型中，准确性的评估能力较差。例如假设我们试图预测一种非常罕见的癌症，这种癌症发生在 0.1% 的人群中。在训练我们的模型后，我们发现准确率为 95%。然而，99.9% 的人没有患上癌症：如果我们简单地创建一个模型来“预测”没有人患上这种癌症，那么我们的幼稚模型的准确率会提高 4.9%，但显然无法预测任何事情。

evaluateClassifier.py

```python
# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成 features matrix and target vector
X, y = make_classification(n_samples=10000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=2,
                           random_state=1)
# 创建 logistic regression
logit = LogisticRegression()
# CV打分函数，scoring为accuracy
# 与原书不同，现在cv函数打分的默认k值改成了5，所以数组有5个元素
print(cross_val_score(logit, X, y, scoring="accuracy"))
```

![image-20220721230843129](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721230843129.png)	



- precision：实际预测正确的阳性观察结果在预测正确中的比例。

  - $Precision=\frac{TP}{TP+FP}$

  - high-precision模型是悲观的，因为它们只有在非常确定的时候才认为观测结果是阳性

    ```python
    # CV 打分函数 scoring为 precision
    print(cross_val_score(logit, X, y, scoring="precision"))
    ```

    

  ![image-20220721231443766](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721231443766.png)

- recall:预测为阳性的结果在真正为阳性的比率。

  - $Recall=\frac{TP}{TP+FN}$

  - 衡量模型识别正类观察的能力。

  - high-recall的模型是乐观的，因为他们把一个类判断成阳性的标准较低

    ```python
    # CV 打分函数 scoring为 recall
    print(cross_val_score(logit, X, y, scoring="recall"))
    ```

    ![image-20220721232753026](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721232753026.png)

- 我们希望precision和recall间能有某种平衡：`F1`

  - F1 分数是调和平均值

  - $F_1=2\times \frac{Precision\times Recall}{Precision+Recall}$

  - ```python
    # CV 打分函数 scoring为 f1
    print(cross_val_score(logit, X, y, scoring="f1"))
    ```

    

![image-20220721232805387](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721232805387.png)

#### Discussion

- 作为一种评估指标，准确率具有一些有价值的特性，尤其是其简单的直觉。

- 更好的指标通常涉及使用精确度和召回率的某种平衡——即我们模型的乐观和悲观之间的权衡。 F1 代表召回率和准确率之间的平衡，其中两者的相对贡献相等。

- 作为使用 cross_val_score 的替代方法，如果我们已经有了真实的 y 值和预测的 y 值，我们可以直接计算准确度和召回等指标：

  ```python
  # Load library
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  
  # 创建测试集和训练集
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.1,
                                                      random_state=1)
  # 对目标向量作出预测
  y_hat = logit.fit(X_train, y_train).predict(X_test)
  # Calculate accuracy
  print(accuracy_score(y_test, y_hat))
  ```

  ![image-20220721233042712](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721233042712.png)



### 11.5 Evaluating Binary Classifier Thresholds

- 需要评估分类器以及各种阈值

#### Solution:`ROC`曲线

- Receiving Operating Characteristic (ROC) curve 是评估二元分类器质量常用的方法。
- ROC 比较每个概率阈值（即，将观察预测为一个类别的概率）处的真阳性和假阳性的存在。
- 绘制 ROC 曲线，我们可以看到模型的表现。 正确预测每个观察结果的分类器看起来像下图中的浅灰色实线，立即直上顶部。 随机预测的分类器将显示为对角线。
- 绘制 ROC 曲线，我们可以看到模型的表现。 正确预测每个观察结果的分类器看起来像下图中的浅灰色实线，立即直上顶部。 随机预测的分类器将显示为对角线。
- 在 scikit-learn 中，我们可以使用 roc_curve 计算每个阈值的真假阳性，然后绘制它们：

rob_curveExample.py

```python
# Load libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# 创建特征矩阵和目标向量
features, target = make_classification(n_samples=10000,  # 10000个样本
                                       n_features=10,  # 10个特征
                                       n_classes=2,  # 2个类别
                                       n_informative=3,  # 参与建模的特征数为3个
                                       random_state=3)  # 随机种子
# 分离训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)
# 创建LogisticRegression 分类器
logit = LogisticRegression()
# 训练模型
logit.fit(features_train, target_train)
# 预测结果
target_probabilities = logit.predict_proba(features_test)[:, 1]
# 错误和正确的比例
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
                                                               target_probabilities)
# 使用pyplot绘制ROC曲线图
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

```

![image-20220721234116265](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721234116265.png)

#### Discussion

- 到目前为止，我们只检查了基于它们预测的值的模型。 然而，在许多学习算法中，这些预测值是基于概率估计的。 也就是说，每个观察都被赋予了属于每个类别的明确概率。
- 我们可以使用 `predict_proba` 查看第一次观察的预测概率：

```python
# 获取第一个observation的分类概率
print(logit.predict_proba(features_test)[0:1])
```

- 预测的类别`classes_`

  ```python
  # 预测的结果
  print(logit.classes_)
  ```

  ![image-20220721234717270](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721234717270.png)

- 第一次观察有大约 87% 的机会属于阴性类 (0)，有 13% 的机会属于阳性类 (1)。默认情况下，如果概率大于 0.5（称为阈值），scikit-learn 会预测观察结果是阳性类的一部分。

- 出于实质性原因，我们通常希望明确地偏向我们的模型以使用不同的阈值，而不是中间立场。例如，如果误报对我们公司来说代价高昂，我们可能更喜欢具有高概率阈值的模型。

____



- 当一个观察结果被预测为阳性时，我们可以非常确信预测是正确的。这种权衡以真阳性率 (TPR) 和假阳性率 (FPR) 表示。真阳性率是正确预测为真的观察值除以所有真阳性观察值：

  - $TPR=\frac{TP}{TP+FN}$

  - $FPR=\frac{FP}{TN+FP}$

  - ROC 曲线代表每个概率阈值的相应 TPR 和 FPR。 例如，在我们的解决方案中，大约 0.50 的阈值具有 0.81 的 TPR 和 0.15 的 FPR

  - ```python
    # 阈值大约为0.5
    print("Threshold:", threshold[116])
    print("True Positive Rate:", true_positive_rate[116])
    print("False Positive Rate:", false_positive_rate[116])
    ```

    ![image-20220721235446694](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721235446694.png)

  - 然而，如果我们将阈值提高到约 80%（即，提高模型在预测观察结果为正之前必须具有的确定性），TPR 会显着下降，但 FPR 也会下降：

  ```python
  # 阈值提升到0.8
  print("Threshold:", threshold[45])
  print("True Positive Rate:", true_positive_rate[45])
  print("False Positive Rate:", false_positive_rate[45])
  ```

  ![image-20220721235918498](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220721235918498.png)

  - 我们对被预测为阳性类的更高要求使得模型无法识别多个阳性样本（较低的 TPR），但也减少了来自被预测为阳性，实际为阴性样本的噪声（较低的 FPR）。

- 除了能够可视化 TPR 和 FPR 之间的权衡之外，ROC 曲线还可以用作模型的通用度量。 模型越好，曲线就越高，因此曲线下的面积就越大。

- 通常通过计算 ROC 曲线下面积 (AUCROC) 来判断模型在所有可能阈值下的整体相等性。 AUCROC 越接近 1，模型越好。 在 scikit-learn 中，我们可以使用 roc_auc_score 计算 AUCROC：

  ```python
  # 计算曲线面积
  print(roc_auc_score(target_test, target_probabilities))
  ```

  ![image-20220722000322509](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722000322509.png)



### 11.6 Evaluating Multiclass Classifier Predictions

- 评估多类分类器模型的质量
- `cross-validation`

evaluateMultiClassifier.py

```python
# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 特征矩阵，目标向量
features, target = make_classification(n_samples=10000,
                                       n_features=3,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=3,
                                       random_state=1)
# 创建 logistic regression
logit = LogisticRegression()
# 使用accuracy指标进行判断
# 与原书不同，现在cv函数打分的默认k值改成了5，所以数组有5个元素
print(cross_val_score(logit, features, target, scoring='accuracy'))
```

![image-20220722102941826](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722102941826.png)

#### Discussion

- 当我们有分布平衡的类时（例如，目标向量的每个类中的观察数量大致相等），就像在二元类设置中一样，准确度是评估指标的一个简单且可解释的选择。准确性是正确预测的数量除以观察的数量，并且在多类和二元设置中的效果一样好。然而，当我们有不平衡的类（一种常见情况）时，我们应该倾向于使用其他评估指标。

- 许多 scikit-learn 的内置指标用于评估二元分类器。

- 但是，当我们有两个以上的类时，可以扩展这些指标中的许多指标以供使用。精度、召回率和 F1 分数是我们在之前的秘籍中已经详细介绍过的有用指标。虽然它们最初都是为二元分类器设计的，但我们可以通过将我们的数据视为一组二元类来将它们应用于多类设置。

  ```python
  # Cross-validate模型 使用 macro averaged F1 score
  print(cross_val_score(logit, features, target, scoring='f1_macro'))
  ```

  ![image-20220722103512422](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722103512422.png)

- 这里的scoring函数后面带着_macro其实是用于平均类评估参数的方法，下面是比较常见的多类评估后缀：

  - macro：

    计算每个类的度量分数的平均值，对每个类进行平均加权。（平均值）

  - weighted：

    计算每个类的度量分数的平均值，根据数据中的大小对每个类进行加权。

  - micro

    计算每个类的度量分数的平均值，根据数据中的大小对每个类进行加权。

- 通俗的来说其实是：

  - ![image-20220722104134857](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722104134857.png)

  - ![image-20220722104200686](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722104200686.png)

  - weighted其实就是对macro中Precision和Recall按照加权平均值来计算，具体就是：

    ![image-20220722104502640](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722104502640.png)

- 参考资料：[(97条消息) 多分类算法的评估指标_taon1607的博客-CSDN博客_多分类算法评价标准](https://blog.csdn.net/taon1607/article/details/107087680)





### 11.7 Visualizing a Classifier’s Performance

- 分类结果可视化
- 绘制混淆矩阵（confusion matrix）

confusionMatrix.py

seaborn库需要安装

- seaborn库介绍：[功能强大的python包（三）：Seaborn - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/389918988)
- seaborn其实就是一个基于matplotlib和pandas的绘制库

```python
conda install seaborn
```



```python
# Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# 莺尾花数据集
iris = datasets.load_iris()
# 特征矩阵
features = iris.data
# 特征向量
target = iris.target
# 创建一个数组包含所有类别的名字
class_names = iris.target_names
# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)
# sigmoid函数回归
classifier = LogisticRegression()
# 训练并且预测
target_predicted = classifier.fit(features_train,
                                  target_train).predict(features_test)
# confusion_matrix创建混淆矩阵
matrix = confusion_matrix(target_test, target_predicted)
# 创建 pandas dataframe（绘制需要是pandas的格式）
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# 创建heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

```

![image-20220722110357382](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722110357382.png)

#### Discussion

- 混淆矩阵是分类器性能的简单、有效的可视化。
- 矩阵的每一列（通常可视化为heatmap)
  - heatmap是常见的一种可视化运用到机器学习中，seaborn也有相关的库函数创建和编辑它
  - 数字越大颜色越深
  - [关于heatmap - 简书 (jianshu.com)](https://www.jianshu.com/p/9fd1b1aee87a)
- 关于混淆矩阵，有三件事值得注意。
  - 一个完美的模型将沿对角线具有值，并且在其他任何地方都具有零。一个糟糕的模型统计的样本数量都集中在边缘的单元格中。
  - 混淆矩阵让我们不仅可以看到模型哪里出错了，还可以看到它是如何出错的（不在对角线上的元素）
  - 混淆矩阵适用于任意数量的类（尽管如果我们的目标向量中有 100 万个类，混淆矩阵的可视化可能难以阅读）。

- 关于sklearn.metrics中的混淆矩阵相关的函数：[sklearn.metrics.confusion_matrix — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)





### 11.8 Evaluating Regression Models

- 评估回归模型
- `mean squared error (MSE)`,中文翻译应该是均方误差

MSE.py

```python
# Load libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 生成 features matrix, target vector
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   n_informative=3,
                                   n_targets=1,
                                   noise=50,
                                   coef=False,
                                   random_state=1)
# 创建线性回归模型
ols = LinearRegression()
# 交叉检验法 linear regression 使用 (negative) MSE
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))
# 交叉检验法 linear regression 使用 R方
print(cross_val_score(ols, features, target, scoring='r2'))
```

![image-20220722111904630](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722111904630.png)

#### Discussion

- MSE 是回归模型最常用的评价指标之一。
  - $MSE = \frac{1}{n}\Sigma_{i=1}^2(\hat y_i-y_i)^2$
- 其中 n 是观察次数，yi 是我们试图预测的目标的真实值，用于观察 i，并且是模型对 yi 的预测值。
- MSE 是预测值和真实值之间所有距离的平方和的度量。 MSE 的值越高，总平方误差越大，因此模型越差。
- **默认情况下，在 scikit-learn 中，评分参数的参数假定较高的值优于较低的值。 但是，对于 MSE，情况并非如此，更高的值意味着更差的模型。 出于这个原因，scikit-learn 使用 neg_mean_squared_error 参数查看负 MSE。**
- 一个常见的替代回归评估指标是 R2 ，它测量模型解释的目标向量中的方差量。（参考11.2节）
  - $R^2 = 1- \frac{\Sigma_i(y_i-\hat{y_i})}{\Sigma_i(y_i-\overline{y})}$





### 11.9 Evaluating Clustering Models

- 评估一个无监督聚类模型的质量
- `silhouette coefficients(轮廓系数)`

silhouetteCoeff.py

```python
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate 特征矩阵
features, _ = make_blobs(n_samples=1000,
                         n_features=10,
                         centers=2,
                         cluster_std=0.5,  # 聚类的标准差
                         shuffle=True,
                         random_state=1)
# 使用 k-means 去预测分类（第19章介绍）
model = KMeans(n_clusters=2, random_state=1).fit(features)
# 获得分完类预测所得到的标签
target_predicted = model.labels_
# 使用silhouette coefficients 预测
print(silhouette_score(features, target_predicted))
```



#### Discussion

- 聚类预测是无监督模型，特点在于没有目标向量，如果没有目标向量，我们无法评估预测值与真实值
- 但我们可以评估集群本身的性质。
- 我们可以想象“好”集群在同一集群中的观测值之间的距离非常小（即密集集群），而不同集群之间的距离很大（即分离良好的集群）。
- `silhouette coefficients`是一个同时衡量这两个属性的单一值
  - $s_i= \frac{b_i-a_i}{max(a_i,b_i)}$其中 si 是观测 i 的轮廓系数，ai 是 i 与同一类的所有观测之间的平均距离，bi 是 i 与来自不同类的最近聚类的所有观测之间的平均距离。
- `silhouette_score `返回的值是所有observation的平均轮廓系数。轮廓系数介于 –1 和 1 之间，其中 1 表示密集、分离良好的集群。
  - api：[sklearn.metrics.silhouette_score — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)



### 11.10 Creating a Custom Evaluation Metric

- 创建自定义的度量方法
- `make_scorer`

make_scorer.py

```python
# Load libraries
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# 生成特征矩阵和目标向量
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   random_state=1)
# 生成测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.10, random_state=1)


# 创建一个自定义的评估度量函数
def custom_metric(target_test, target_predicted):
    # 计算r2_score
    r2 = r2_score(target_test, target_predicted)
    # 返回
    return r2


# 定义scorer，然后greater_is_better表明分数越高模型越好
score = make_scorer(custom_metric, greater_is_better=True)
# 脊回归
classifier = Ridge()
# 训练脊回归模型
model = classifier.fit(features_train, target_train)
# 使用自定义的scorer
print(score(model, features_test, target_test))
```



![image-20220722114759142](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722114759142.png)

#### Discussion

- make_scorer需要我们定义一个函数f：

  - f接受两个参数：ground_truth(目标向量)、predicted（我们的预测值）

- make_scorer需要`greater_is_better`指定是否是分越高越好

- 在案例中我们将R2假装成我们自定义的判断方法来作为

  ```python
  # 预测值
  target_predicted = model.predict(features_test)
  # 使用r2_score评分
  print(r2_score(target_test, target_predicted))
  ```

  ![image-20220722115425666](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722115425666.png)

- 关于示例出现的Ridge回归：（中文翻译用岭回归也有翻译成脊回归的）
  - Ridge是线性回归的一种改良，在线性回归的基础上增加一个正则项（惩罚项）
  - 当X不是列满秩时，或者某些列之间的线性相关性比较大时，
    $X^TX$的行列式接近于0，不能采用最小二乘法进行求解。
  - 所以我们添加一个正则项，$\Gamma\theta$
  - 讲的比较详细的资料：[简单易学的机器学习算法——岭回归(Ridge Regression) - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1390622)



### 11.11 Visualizing the Effect of Training Set Size

- 评估Observation的数量对模型质量的影响

- 绘制`learning curve`

learningCurve.py

```python
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# 手写数字集
digits = load_digits()
# 特征矩阵和目标向量
features, target = digits.data, digits.target
# 创建多种数据集大小对应的cv检验的结果
train_sizes, train_scores, test_scores = learning_curve(
    # 分类器——随机森林
    RandomForestClassifier(),
    # 特征矩阵
    features,
    # 目标向量
    target,
    # Kfolds
    cv=10,
    # 评估函数
    scoring='accuracy',
    # 所有CPU参与评估
    n_jobs=-1,
    # 大小为50
    # 训练集
    train_sizes=np.linspace(
        0.01, #百分之1
        1.0,  #百分百
        
        50))
# 训练集平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# 测试集的平均值和标准差
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# 绘制条带
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")
# 坐标系
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()

```

![image-20220722144605845](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722144605845.png)

#### Discussion

- 随着训练集中观察数量的增加，学习曲线可视化模型在训练集和交叉验证期间的性能（例如，准确性、召回率）。 它们通常用于确定我们的学习算法是否会从收集额外的训练数据中受益。
- 我们绘制了随机森林分类器在 50 个不同的训练集大小（从 1% 到 100% 的观测值）下的准确度。 交叉验证模型的准确性得分不断提高告诉我们，我们可能会从额外的观察中受益（尽管在实践中这可能不可行）。



### 11.12 Creating a Text Report of Evaluation Metrics

- 输出模型评估结果的基本信息
- `classification_report`

textReportofEvaluation.py

```python
# Load libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载莺尾花数据集
iris = datasets.load_iris()
# 特征矩阵
features = iris.data
# 目标向量
target = iris.target
# 类的名字
class_names = iris.target_names
# 训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)
# 创建 logistic regression
classifier = LogisticRegression()
# 训练模型并作出预测
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)
# 创建分类器评估结果的简短描述
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))

```

![image-20220722145603676](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722145603676.png)

#### Discussion

- Classification_report 为我们提供了一种快速查看一些常见评估指标的方法
  - precision
  - recall
  - F1-score 
  - support：每个类的observation的个数



### 11.13 Visualizing the Effect of Hyperparameter Values

- 评估不同超参数对模型的影响

- 绘制`validation curve`

  validationCurve.py

  ```python
  # Load libraries
  import matplotlib.pyplot as plt
  import numpy as np
  from sklearn.datasets import load_digits
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import validation_curve
  
  # 手写数字集
  digits = load_digits()
  # 特征矩阵，目标矩阵
  features, target = digits.data, digits.target
  # 生成一个数组从1到250，步长为2作为超参数数组
  param_range = np.arange(1, 250, 2)
  # 通过不同的参数生成training and test
  train_scores, test_scores = validation_curve(
      # 随机森林的分类器
      RandomForestClassifier(),
      # 特征矩阵
      features,
      # 目标向量
      target,
      # 超参数
      param_name="n_estimators",
      # 超参数的范围
      param_range=param_range,
      # KFold的k值
      cv=3,
      # 评估标准
      scoring="accuracy",
      # 使用所有CPU
      n_jobs=-1)
  # 计算训练集的平均值和标准差
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  # 计算测试集的平均值和标准差
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  # 绘制accuracy
  plt.plot(param_range, train_mean, label="Training score", color="black")
  plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
  # 绘制accuracy的条带
  plt.fill_between(param_range, train_mean - train_std,
                   train_mean + train_std, color="gray")
  plt.fill_between(param_range, test_mean - test_std,
                   test_mean + test_std, color="gainsboro")
  # 绘制
  plt.title("Validation Curve With Random Forest")
  plt.xlabel("Number Of Trees")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.show()
  
  ```

  

![image-20220722151535724](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722151535724.png)

#### Discussion

- 大多数训练算法都必须在训练前选择自己的超参数
  - 本案例中我们使用的随机森林算法（第14章会介绍），其中的一个超参数就是森林中决策树的数量
  - 大多数超参数是在模型选择（Model Selection)的阶段进行被选择的
  - 我们通过`validation curve`来展示了不同决策树数量对于最后准确度的影响
- 当我们有少量的树时，训练和交叉验证的分数都很低，这表明模型拟合不足。
- 随着树的数量增加到 250 棵，两者的准确性都趋于平稳，这表明训练大规模森林的计算成本可能没有太大价值。
- `validation_curve `
  - `param_name`:超参数的名称
  - `param_range`:超参数的范围
  - `scoring`：评估的方式和之前一样：accuracy、precision等等





## Chapter 12. Model Selection

### 12.0 Introduction

- 在机器学习中，我们使用训练算法通过最小化一些损失函数来学习模型的参数。除此之外，许多学习算法（例如，支持向量分类器和随机森林）也有必须在学习过程之外定义的**超参数（hyperparameters）**。
- 我们通常可能想尝试多种学习算法（例如，同时尝试支持向量分类器和随机森林，看看哪种学习方法产生了最好的模型）。
- 我们将选择最佳学习算法及其最佳超参数都称为**模型选择**。
- 在本章中，我们将介绍从候选集中有效地选择最佳模型的技术。
- 在本章中，我们将参考特定的超参数。，例如 C（正则化强度的倒数）。





### 12.1 Selecting Best Models Using Exhaustive Search

- 通过穷举法来找出最好的模型
- `GridSearchCV`

exhaustiveSearch.py

```python
# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# 莺尾花数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建 logistic regression
logistic = linear_model.LogisticRegression()
# 创造超参数-regularization penalty的可能的序列
penalty = ['l1', 'l2']
# 创建C的可能的序列
C = np.logspace(0, 4, 10)  # np.logspace生成等比数列
# 创建一个字典，C和penalty分别指向两个参数
hyperparameters = dict(C=C, penalty=penalty)
# 查看参数信息
print(hyperparameters)
# 进行穷举搜索
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# fit最佳模型
best_model = gridsearch.fit(features, target)

```



#### Discussion

- `GridSearchCV` 是一种使用交叉验证进行模型选择的暴力方法。

  - 用户为一个或多个超参数定义一组可能的值，然后 GridSearchCV 使用每个值和/或值的组合训练模型。
  - 选择性能得分最高的模型作为最佳模型。

- 解析一下我们代码是怎么寻找到最佳模型

  - 在我们的解决方案中，我们使用**逻辑回归（LogisticRegression）**（将在接下来的章节介绍，所以并不需要掌握C和正则化惩罚参数是什么，只需要知道他们是超参数）作为我们的模型

  - 逻辑回归拥有两个超参数：

    - C
    -  regularization penalty

  - 对于我们的C我们使用numpy的`logspace`创建了一组等比数列

    ```python
    np.logspace(0, 4, 10)
    ```

    ![image-20220722160830777](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722160830777.png)

  - 同样我们也定义了两个正则化惩罚可能的值：[l1,l2]

  - 检验方法我们选择的是k值为5的k折交叉检验法

  - 而`GridSearchCV`暴力的创建了10（C值的个数）× 2（正则化惩罚的个数）×5（k折交叉检验）个候选模型，从这100个模型里选择出评估得分最高的

  - 我们可以查看最好模型的超参数，并且使用它进行预测：

    ```python
    # 查看超参数
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    # 预测
    print(best_model.predict(features))
    ```

    

![image-20220722161311708](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722161311708.png)

- `GridSearchCV`的参数：
  - verbose：最值得注意的一个参数：verbose 参数决定了搜索过程中输出的消息量，0 表示没有输出，1 到 3 表示输出的消息越来越详细
  - cv：交叉检验法
  - n_job、scoring和之前的参数一样
  - api:[sklearn.model_selection.GridSearchCV — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)



### 12.2 Selecting Best Models Using Randomized Search

- 对模型进行随机搜索
- `RandomizedSearchCV`

randomizedSearch.py 

```
# Load libraries
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# 加载莺尾花
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression()
# 惩罚项可能的值
penalty = ['l1', 'l2']
# C可能的值
C = uniform(loc=0, scale=4)  # 随机数生成C
# 创建超参数字典供searchCv选择
hyperparameters = dict(C=C, penalty=penalty)
# 随机化搜索
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)
# 选择出最好的模型并训练
best_model = randomizedsearch.fit(features, target)

```



#### Discussion

- `RandomizedSearchCV`的原理是从用户提供的分布（例如，正态分布、均匀分布）中搜索特定数量的超参数值的随机组合。

- 如果我们指定一个分布，scikit-learn 将随机抽样而不像`GridSearchCV`从该分布中替换超参数值。

  - 本示例中，我们从 0 到 4 的均匀分布中随机抽取 10 个值作为C的候选序列：

  ![image-20220722162953333](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722162953333.png)

  - 我们和上一节一样，以[l1,l2]作为惩罚项的候选序列，但是本例中RandomizedSearchCV不是生成更多的模型，而是对两者进行随机的抽样

- 像使用 GridSearchCV 一样，我们可以看到最佳模型的超参数值：

- 最佳模型也可以进行预测

  ```python
  # 查看超参数
  print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
  print('Best C:', best_model.best_estimator_.get_params()['C'])
  
  # 预测目标
  best_model.predict(features)
  ```

  ![image-20220722163622615](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722163622615.png)

- 超参数的采样组合的数量（即训练的候选模型的数量）由 `n_iter`（迭代次数）设置指定。
- api:[sklearn.model_selection.RandomizedSearchCV — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)



### 12.3 Selecting Best Models from Multiple Learning Algorithms

- 从多种学习算法中选择出最佳模型
- 建立一个包含多种学习算法和它们各自的参数的字典

multiAlgorithm.py

```python
# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 创建随机数种子
np.random.seed(0)
# 加载莺尾花数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建一个管道进行训练优化
pipe = Pipeline([("classifier", RandomForestClassifier())])
# 创建一个字典，包含学习算法数组和他们的参数
search_space = [{"classifier": [LogisticRegression()],  # 逻辑回归
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],  # 随机森林
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]
# 穷举搜索和cv交叉检验评估
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# 选择出的模型进行训练
best_model = gridsearch.fit(features, target)

```



#### Discussion

- 我们可以通过字典扩大搜索空间，从而实现从多种学习算法中选择

- 本例中我们在LogisticRegression和RandomForestClassifier中进行选择

- 搜索结束后可以查看选择的最佳模型的学习算法，超参数等信息

  ```python
  # 查看模型
  print(best_model.best_estimator_.get_params()["classifier"])
  # 进行预测
  print(best_model.predict(features))
  ```

  

![image-20220722164750102](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722164750102.png)





### 12.4 Selecting Best Models When Preprocessing

- 在模型选择的过程中进行数据预处理
- 创建pipeline并将预处理加入到pipeline中



preprocessing.py

```python
# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 随机数种子
np.random.seed(0)
# 加载莺尾花数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 预处理包括预处理和PCA降维
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])
# 创建一个管道包含预处理和模型选择
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])
# PCA参数的搜索空间和超参数的搜索空间
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]
# 暴力搜索
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
# 训练
best_model = clf.fit(features, target)

```



#### Discussion

- 很多时候，我们需要在使用它来训练模型之前对数据进行预处理。

- 在进行模型选择时，我们必须小心正确地处理预处理.

  - 首先，GridSearchCV 使用交叉验证来确定哪个模型具有最高性能.
  - 在交叉验证中，我们实际上是在假装折叠保持不变，因为没有看到测试集，因此不是拟合任何预处理步骤（例如，缩放或标准化）的一部分。出于这个原因，我们不能预处理数据然后运行 GridSearchCV。

- `FeatureUnion` 允许我们正确地组合多个预处理操作。在我们的解决方案中，我们使用 `FeatureUnion` 来组合两个预处理步骤：标准化特征值（`StandardScaler`(第4章)）和主成分分析（PCA（第9章））。

- 我们使用我们的学习算法将预处理包含到管道中。最终结果是，这使我们能够将使用超参数组合的模型的拟合、转换和训练的正确（和令人困惑的）处理外包给 scikit-learn。

- 一些预处理方法有自己的参数，这些参数通常必须由用户提供。例如，使用 PCA 进行降维需要用户定义用于生成转换特征集的主成分的数量。scikit-learn 让这一切变得简单。当我们在搜索空间中包含候选组件值时，它们被视为要搜索的任何其他超参数。

- 模型选择完成后，我们可以查看产生最佳模型的预处理值。

  ```python
  # 最佳模型的PCA特征数量
  print(best_model.best_estimator_.get_params()['preprocess__pca__n_components'])
  ```

  

![image-20220722170138390](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722170138390.png)

### 12.5 Speeding Up Model Selection with Parallelization

- 加速模型选择
- `n_jobs=-1`

speedingUp.py

```python
# Load libraries

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
import datetime
starttime = datetime.datetime.now()


# 加载数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression()
# penalty超参数候选值
penalty = ["l1", "l2"]
# C候选值
C = np.logspace(0, 4, 1000)
# 创建超参数搜索空间
hyperparameters = dict(C=C, penalty=penalty)
# 暴力搜索
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)
# 训练模型
best_model = gridsearch.fit(features, target)

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
```

![image-20220722171122035](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722171122035.png)

运行48s

将`n_jobs`改为1

![image-20220722171424634](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722171424634.png)

时间为155s

#### Discussion

- 在现实世界中，我们通常会有数千或数万个模型需要训练。最终结果是找到最佳模型可能需要花费数小时
- 为了加快这个过程，scikit-learn 让我们可以同时训练多个模型。在不涉及太多技术细节的情况下，scikit-learn 可以同时训练模型达到机器上的核心数量。
- 参数 n_jobs 定义要并行训练的模型数量。在我们的解决方案中，我们将 n_jobs 设置为 -1，这告诉 scikit-learn 使用所有内核。
- 默认情况下 n_jobs 设置为 1，这意味着它只使用一个核心。



### 12.6 Speeding Up Model Selection Using Algorithm-Specific Methods

- 和上一节的目标一样，我们需要加速模型选择

- 假设需要在特定的学习方法中选择模型，使用 scikit-learn中模型交叉验证的超参数进行调整。

  例如LogisticRegressionCV:

  specificMethods.py

```python
# Load libraries
from sklearn import linear_model, datasets
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100)
# Train model
print(logit.fit(features, target))
```

![image-20220722204100510](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722204100510.png)

### Discussion

- 在 scikit-learn 中，许多学习算法（例如 Ridge回归、lasso回归 和elastic net regression（弹性网络回归算法））都有一种特定于该算法的交叉验证方法

  - 例如，LogisticRegression 用于进行标准逻辑回归分类器，而 LogisticRegressionCV 实现了一个高效的交叉验证逻辑回归分类器，能够识别超参数 C 的最佳值。

  - 参数`CS`:C的一系列候选值

    - 如果是列表，则Cs作为一个超参数，列表中的值是Cs的候选值
    - 如果提供列表，则 Cs 是要从中选择的候选超参数值。
    - 候选值是从 0.0001 到 1,0000 之间的范围（C 的合理值范围）以对数方式得出的。

  - LogisticRegressionCV 的一个主要缺点是它只能搜索 C 的一系列值。在 12.1 节中，我们可能的超参数空间包括 C 和另一个超参数（正则化惩罚范数）。

    **这样无法照顾到全部超参数的限制在 scikit-learn 的许多特定于模型的交叉验证方法中很常见**

#### scikit-learn常见的特定交叉验证方法：

[3.2. Tuning the hyper-parameters of an estimator — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation)



### 12.7 Evaluating Performance After Model Selection

- 在选择模型之后评估模型的质量
- 使用嵌套的交叉验证评估来避免评估具有偏差

evaluateAfterSelecting.py

```python
# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression()
# 创建20个候选的C值
C = np.logspace(0, 4, 20)
# 可选择的超参数的代数空间
hyperparameters = dict(C=C)
# 穷举搜索
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)
# 嵌套的交叉检验计算的出平均值
print(cross_val_score(gridsearch, features, target).mean())
```



![image-20220722221216223](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722221216223.png)

#### Discussion

- 由于我们已经使用了交叉检验来产生了最佳的模型，但是如果我们还使用同样的数据来进行评估的话，结果明显是不可靠的。

- 因此产生了嵌套交叉检验的方法。“内部”交叉验证选择最佳模型，而“外部”交叉验证为我们提供了对模型性能的无偏见评估。

- 在我们的解决方案中，内部交叉验证是我们的` GridSearchCV `对象，然后我们使用` cross_val_score` 将其包装在外部交叉验证中。

- 可能这样比较晦涩，在前几节中我们学习了`verbose`参数可以控制输出的信息。

  - 我们使用verbose=1：

    ```python
    gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)
    ```

  - 运行训练最佳模型的fit，生成一条信息（内部交叉检验产生的）

    ```python
    # 查看嵌套时的信息
    # 内部
    best_model = gridsearch.fit(features, target)
    ```

    得到结果：

    ![image-20220722223231761](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722223231761.png)

  - 运行`cross_val_score`:

  ```python
  # 外部
  scores = cross_val_score(gridsearch, features, target)
  ```

  ![image-20220722223439419](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20220722223439419.png)

  生成的结果可以看到内部的CV又训练了5次100的模型

  - 我们可以从结果中看出，cross_val_score需要进行五折交叉检验（原书为旧版本scikit-learn,默认为3折)，然后内层的每次需要进行5折的交叉检验，所以嵌套的交叉检验总共要进行20*5\*5=500次训练

