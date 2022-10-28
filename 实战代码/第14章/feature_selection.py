# 库
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

iris = datasets.load_iris()
features = iris.data
target = iris.target
# 随机僧林分类器
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# 设置阈值
selector = SelectFromModel(randomforest, threshold=0.3)
# 新的特征矩阵
features_important = selector.fit_transform(features, target)
# 训练
model = randomforest.fit(features_important, target)
