# 加载库
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

# 加载加州房价
housing = fetch_california_housing()
features = housing.data[:, 0:1]
target = housing.target
# 多项式 x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
# 创建线性关系
regression = LinearRegression()
# 拟合线性关系
model = regression.fit(features_polynomial, target)

print(features[0])
print(features[0]**2)
print(features[0]**3)
print(features_polynomial[0])
