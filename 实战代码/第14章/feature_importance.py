
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# 莺尾花数据集
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 随机森林分类器
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# 模型训练
model = randomforest.fit(features, target)
# 计算特征的重要程度
importances = model.feature_importances_
# 排序
indices = np.argsort(importances)[::-1]
# 重排
names = [iris.feature_names[i] for i in indices]


plt.figure()
plt.title("Feature Importance")
plt.bar(range(features.shape[1]), importances[indices])
# 添加特征名到X轴
plt.xticks(range(features.shape[1]), names, rotation=0)
# 显示图
plt.show()
