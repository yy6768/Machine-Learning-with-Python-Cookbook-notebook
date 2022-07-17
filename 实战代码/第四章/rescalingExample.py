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
