# 加载库
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# 加载加州房价
housing = fetch_california_housing()
features = housing.data[:,0:2]
target = housing.target
# 创建线性回归模型
regression = LinearRegression()
# 在线性模型中获取均值和方差
model = regression.fit(features, target)

# 查看差值
print(model.intercept_)
# 查看系数集合
print(model.coef_)
# 进行预测
print(model.predict(features)[0])
