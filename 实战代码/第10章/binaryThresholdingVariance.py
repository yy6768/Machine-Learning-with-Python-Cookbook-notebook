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