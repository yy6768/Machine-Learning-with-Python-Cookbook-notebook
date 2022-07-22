# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# 莺尾花数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建 logistic regression
logistic = linear_model.LogisticRegression()
# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']
# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)  # np.logspace生成等比数列
# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)
print(hyperparameters)
# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = gridsearch.fit(features, target)

# 查看超参数
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# 预测
print(best_model.predict(features))