# Load libraries
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
# 创建一个线性可分的数据 圆数据集
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# 使用 kernal PCA  核函数RBF 系数15，降维到1维
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])