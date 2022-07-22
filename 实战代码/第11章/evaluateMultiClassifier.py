# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 特征矩阵，目标向量
features, target = make_classification(n_samples=10000,
                                       n_features=3,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=3,
                                       random_state=1)
# 创建 logistic regression
logit = LogisticRegression()
# 使用accuracy指标进行判断
# 与原书不同，现在cv函数打分的默认k值改成了5，所以数组有5个元素
print(cross_val_score(logit, features, target, scoring='accuracy'))

# Cross-validate模型 使用 macro averaged F1 score
print(cross_val_score(logit, features, target, scoring='f1_macro'))