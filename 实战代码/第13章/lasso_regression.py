# 加载python库
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
# 加利福尼亚房价数据集
housing = fetch_california_housing()
features = housing.data
target = housing.target
# 标准化数据
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

# View coefficients
print(model.coef_)

regression_10 = Lasso(alpha=10)
model = regression_10.fit(features_standardized, target)
print(model.coef_)