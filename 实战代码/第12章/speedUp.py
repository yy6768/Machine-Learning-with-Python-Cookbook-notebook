# Load libraries

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
import datetime
starttime = datetime.datetime.now()


# 加载数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression()
# penalty超参数候选值
penalty = ["l1", "l2"]
# C候选值
C = np.logspace(0, 4, 1000)
# 创建超参数搜索空间
hyperparameters = dict(C=C, penalty=penalty)
# 暴力搜索
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)
# 训练模型
best_model = gridsearch.fit(features, target)

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)