# 库
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets

# 数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 高度不平衡的数据（让除了0以外的类为1类）
test_features = features[0:40, :]
features = features[40:, :]
test_target = target[0:40]
target = target[40:]

# 除了0其他都是1
test_target = np.where((target == 0), 0, 1)
target = np.where((target == 0), 0, 1)
# Create random forest classifier object
randomforest = RandomForestClassifier(
    random_state=0, n_jobs=-1, class_weight="balanced")
# 训练模型
model = randomforest.fit(features, target)

randomforest_imbalance = RandomForestClassifier(
    random_state=0, n_jobs=-1
)

model_imbalance = randomforest_imbalance.fit(features, target)
print(model.predict(test_features))
print(model.predict(test_features))
print(test_target)

