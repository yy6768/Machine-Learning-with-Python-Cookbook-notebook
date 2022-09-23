# 加载库
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

# 加载加州房价
housing = fetch_california_housing()
features = housing.data[:, 0:2]
target = housing.target

# 创建交互项
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)
# 创建线性回归模型
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_interaction, target)

# View first observation
print(features[0])
# Import library
import numpy as np
# For each observation, multiply the values of the first and second feature
interaction_term = np.multiply(features[:, 0], features[:, 1])
print(interaction_term[0])