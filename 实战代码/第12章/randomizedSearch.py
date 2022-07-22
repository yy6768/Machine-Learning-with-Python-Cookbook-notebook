# Load libraries
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# 加载莺尾花
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 逻辑回归
logistic = linear_model.LogisticRegression()
# 惩罚项可能的值
penalty = ['l1', 'l2']
# C可能的值
C = uniform(loc=0, scale=4)  # 随机数生成C
# 创建超参数字典供searchCv选择
hyperparameters = dict(C=C, penalty=penalty)
# 随机化搜索
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)
# 选择出最好的模型并训练
best_model = randomizedsearch.fit(features, target)

print(uniform(loc=0, scale=4).rvs(10))

# 查看超参数
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# 预测目标
print(best_model.predict(features))