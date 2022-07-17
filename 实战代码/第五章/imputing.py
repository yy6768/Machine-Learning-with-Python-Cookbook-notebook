# load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0, 2.10, 1.45],
            [1, 1.18, 1.33],
            [0, 1.22, 1.27],
            [1, -0.21, -1.19]])

X_with_nan = np.array([[np.nan, 0.87, 1.31],
                      [np.nan, -0.67, -0.22]])

# 训练 KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:, 0])

# 预测丢失值
imputed_values = trained_model.predict(X_with_nan[:, 1:])

# 连接丢失的列和他们原来其他列
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# 连接原来的矩阵和有丢失列的矩阵
print(np.vstack((X_with_imputed, X)))

# 方法2 用最常用的值去替换缺失值
# impute已经被替代
from sklearn.impute import SimpleImputer
# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
imputer = SimpleImputer(strategy='most_frequent')
print(imputer.fit_transform(X_complete))