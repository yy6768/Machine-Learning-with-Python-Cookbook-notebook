# Load libraries
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(n_samples=1000,
                         n_features=2,
                         random_state=1)
# 标准化
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
# 替换第一个值为NAN
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan
# 预测
features_knn_imputed = KNN(k=5, verbose=0).fit_transform(standardized_features)
# 比较
print("True Value:", true_value)
print("Imputed Value:", features_knn_imputed[0, 0])

# Load library
from sklearn.impute import SimpleImputer

# Create imputer
mean_imputer = SimpleImputer(strategy="mean")
# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])
