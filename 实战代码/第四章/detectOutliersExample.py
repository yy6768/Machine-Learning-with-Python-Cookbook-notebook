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