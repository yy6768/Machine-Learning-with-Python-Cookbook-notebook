# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression(max_iter=1000)
# 创建20个候选的C值
C = np.logspace(0, 4, 20)
# 可选择的超参数的代数空间
hyperparameters = dict(C=C)
# 穷举搜索
# gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)
# 嵌套的交叉检验计算的出平均值
# print(cross_val_score(gridsearch, features, target).mean())

# 查看嵌套时的信息
# 内部
best_model = gridsearch.fit(features, target)
# 外部
scores = cross_val_score(gridsearch, features, target, cv=5, verbose=1)

