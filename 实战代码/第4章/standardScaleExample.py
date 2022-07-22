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