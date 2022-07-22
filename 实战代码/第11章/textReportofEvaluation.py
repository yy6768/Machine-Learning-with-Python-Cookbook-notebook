# Load libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载莺尾花数据集
iris = datasets.load_iris()
# 特征矩阵
features = iris.data
# 目标向量
target = iris.target
# 类的名字
class_names = iris.target_names
# 训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)
# 创建 logistic regression
classifier = LogisticRegression()
# 训练模型并作出预测
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)
# 创建分类器评估结果的简短描述
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))
