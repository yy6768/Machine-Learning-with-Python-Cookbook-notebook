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