# Load libraries
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
# 加利福尼亚房价数据集
housing = fetch_california_housing()
features = housing.data
target = housing.target
# 标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# 岭回归
regression = Ridge(alpha=0.5)
# 适用模型
model = regression.fit(features_standardized, target)


# Load library
from sklearn.linear_model import RidgeCV
# 创建一系列的alpha参数，使用交叉检验法进行比较
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# 拟合模型
model_cv = regr_cv.fit(features_standardized, target)
# 查看结果
print(model_cv.coef_)
print(model_cv.alpha_)
