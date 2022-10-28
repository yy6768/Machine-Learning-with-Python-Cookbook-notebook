# Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 训练一个随机森林算法
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# 训练模型
model = randomforest.fit(features, target)

# 创建新的样本集
observation = [[5, 4, 3, 2]]
# 预测
print(model.predict(observation))

# 使用信息熵来作为指标训练随机森林
random_forest_entropy = RandomForestClassifier(
    criterion="entropy", random_state=0)
# Train model
model_entropy = random_forest_entropy.fit(features, target)


print(model_entropy.predict(observation))