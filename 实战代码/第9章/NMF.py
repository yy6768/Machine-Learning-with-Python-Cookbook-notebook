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