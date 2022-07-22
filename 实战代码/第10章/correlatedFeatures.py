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
print(corr_matrix)
# 选取这个矩阵的上三角部分（因为对称）
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k=1).astype(bool))
# 寻找特征之间线性相关度大于0.95的特征
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# 删除这些特征
print(dataframe.drop(dataframe.columns[to_drop], axis=1).head(3))