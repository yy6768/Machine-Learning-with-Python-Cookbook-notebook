# Load libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# 手写数字集
digits = load_digits()
# 特征矩阵，目标矩阵
features, target = digits.data, digits.target
# 生成一个数组从1到250，步长为2作为超参数数组
param_range = np.arange(1, 250, 2)
# 通过不同的参数生成training and test
train_scores, test_scores = validation_curve(
    # 随机森林的分类器
    RandomForestClassifier(),
    # 特征矩阵
    features,
    # 目标向量
    target,
    # 超参数
    param_name="n_estimators",
    # 超参数的范围
    param_range=param_range,
    # KFold的k值
    cv=3,
    # 评估标准
    scoring="accuracy",
    # 使用所有CPU
    n_jobs=-1)
# 计算训练集的平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# 计算测试集的平均值和标准差
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制accuracy
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
# 绘制accuracy的条带
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, color="gainsboro")
# 绘制
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
