# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 创建随机数种子
np.random.seed(0)
# 加载莺尾花数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建一个管道进行训练优化
pipe = Pipeline([("classifier", RandomForestClassifier())])
# 创建一个字典，包含学习算法数组和他们的参数
search_space = [{"classifier": [LogisticRegression()],  # 逻辑回归
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],  # 随机森林
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]
# 穷举搜索和cv交叉检验评估
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# 选择出的模型进行训练
best_model = gridsearch.fit(features, target)

# 查看模型
print(best_model.best_estimator_.get_params()["classifier"])
# 进行预测
print(best_model.predict(features))