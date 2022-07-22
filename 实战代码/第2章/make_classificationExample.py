# load library
from sklearn.datasets import make_classification

# generate features matrix and target vector

features, target = make_classification(n_samples=100,  # 样本个数
                                       n_features=3,  # 特征数
                                       n_informative=3,  # 参与建模的特征数
                                       n_redundant=0,  # 冗余信息
                                       n_classes=2,  # 类的个数
                                       weights=[.25, .75],  # 权重
                                       random_state=1)  # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature matrix\n {}".format(features[:3]))
print("Target vector\n {}".format(target[:3]))
