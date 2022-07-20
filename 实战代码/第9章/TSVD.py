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

# 打印前3个维度上的方差
print(tsvd.explained_variance_ratio_[0:3].sum())

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
