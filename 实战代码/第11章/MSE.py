# Load libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 生成 features matrix, target vector
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   n_informative=3,
                                   n_targets=1,
                                   noise=50,
                                   coef=False,
                                   random_state=1)
# 创建线性回归模型
ols = LinearRegression()
# 交叉检验法 linear regression 使用 (negative) MSE
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))
# 交叉检验法 linear regression 使用 R方
print(cross_val_score(ols, features, target, scoring='r2'))