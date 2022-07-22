# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 随机数种子
np.random.seed(0)
# 加载莺尾花数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 预处理包括预处理和PCA降维
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])
# 创建一个管道包含预处理和模型选择
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])
# PCA参数的搜索空间和超参数的搜索空间
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]
# 暴力搜索
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
# 训练
best_model = clf.fit(features, target)

# 最佳模型的PCA特征数量
print(best_model.best_estimator_.get_params()['preprocess__pca__n_components'])