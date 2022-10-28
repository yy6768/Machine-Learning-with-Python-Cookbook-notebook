# Load libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

# Load data with only two features
boston = datasets.load_boston()
features = boston.data[:, 0:2]
target = boston.target
# Create decision tree classifier object
decisiontree = DecisionTreeRegressor(random_state=0)
# 训练模型
model = decisiontree.fit(features, target)

# Make new observation
observation = [[0.02, 16]]
# 预测
model.predict(observation)

decisiontree_mae = DecisionTreeRegressor(random_state=0, criterion='mae')
model_mae = decisiontree_mae.fit(features, target)
