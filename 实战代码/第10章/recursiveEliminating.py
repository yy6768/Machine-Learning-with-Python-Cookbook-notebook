# 加载库
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# 丢弃警告
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")
# 生成有10000个样本，100个特征的线性回归的样本
features, target = make_regression(n_samples=10000,
                                   n_features=100,
                                   n_informative=2,
                                   random_state=1)
# 创建一个线性模型
ols = linear_model.LinearRegression()
# 递归的去除特征
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
print(rfecv.transform(features))
# 查看好的特征的数量
print(rfecv.n_features_)

# 查看哪些特征应该被保留
print(rfecv.support_)

# 我们可以查看特征的排行
print(rfecv.ranking_)