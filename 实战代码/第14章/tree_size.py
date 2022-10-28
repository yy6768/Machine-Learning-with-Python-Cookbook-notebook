# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,  # 最大深度
                                      min_samples_split=2,  # 内部有节点最小的样本数
                                      min_samples_leaf=1,  # 叶子节点最小的样本数
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,  # 最大叶子节点数
                                      min_impurity_decrease=0)  # 最小脏度
# Train model
model = decisiontree.fit(features, target)
