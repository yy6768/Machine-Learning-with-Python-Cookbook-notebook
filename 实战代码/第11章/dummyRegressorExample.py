# Load libraries
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
# 特征矩阵、目标
features, target = housing.data, housing.target
# 分离测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)
# 创建一个DummyRegressor
dummy = DummyRegressor(strategy='mean')
# 训练它
dummy.fit(features_train, target_train)
# 得到R^2得分
print(dummy.score(features_test, target_test))

# Load library
from sklearn.linear_model import LinearRegression

# 训练一个简单的线性模型
ols = LinearRegression()
ols.fit(features_train, target_train)
# 得分
print(ols.score(features_test, target_test))

# 每一次都预测为20
clf = DummyRegressor(strategy='constant', constant=2)
clf.fit(features_train, target_train)
# 得分
print(clf.score(features_test, target_test))