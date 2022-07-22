# Load libraries
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# 莺尾花数据
iris = load_iris()
# 创建target vector  feature matrix
features, target = iris.data, iris.target
# 分离训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)
# 创建 dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)
# "训练" model
dummy.fit(features_train, target_train)
# 获得准确性的评分
print(dummy.score(features_test, target_test))


# Load library
from sklearn.ensemble import RandomForestClassifier
# 创建一个随机森林的分类器（14章会介绍原理）
classifier = RandomForestClassifier()
# 训练模型
classifier.fit(features_train, target_train)
# 得到准确性的评分
print(classifier.score(features_test, target_test))