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

# 查看方差
print(thresholder.fit(features).variances_)

