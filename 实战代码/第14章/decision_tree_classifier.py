# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建一棵决策树
decisiontree = DecisionTreeClassifier(random_state=0)
# 训练模型
model = decisiontree.fit(features, target)

# Make new observation
observation = [[5, 4, 3, 2]]
# Predict observation's class预测类
print(model.predict(observation))
# 预测属于各个类的概率
print(model.predict_proba(observation))

# Create decision tree classifier object using 信息熵
decisiontree_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0)
# Train model
model_entropy = decisiontree_entropy.fit(features, target)
