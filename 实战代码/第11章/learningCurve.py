# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# 手写数字集
digits = load_digits()
# 特征矩阵和目标向量
features, target = digits.data, digits.target
# 创建多种数据集大小对应的cv检验的结果
train_sizes, train_scores, test_scores = learning_curve(
    # 分类器——随机森林
    RandomForestClassifier(),
    # 特征矩阵
    features,
    # 目标向量
    target,
    # Kfolds
    cv=10,
    # 评估函数
    scoring='accuracy',
    # 所有CPU参与评估
    n_jobs=-1,
    # 大小为50
    # 训练集
    train_sizes=np.linspace(
        0.01,
        1.0,
        50))
# 训练集平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# 测试集的平均值和标准差
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# 绘制条带
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")
# 坐标系
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()
