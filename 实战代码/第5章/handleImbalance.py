# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
# 加载莺尾花数据集
iris = load_iris()
# 创建特征矩阵
features = iris.data
# 创建目标向量
target = iris.target
# 移除前40个元素
features = features[40:,:]
target = target[40:]
# 将类0和非类0的observation分为两类
target = np.where((target == 0), 0, 1)
# 打印
print(target)

# 第一种处理，提供类权重

# 创建权重
weights = {0: .9, 1: 0.1}
# 根据权重创建随机森岭
print(RandomForestClassifier(class_weight=weights))

# Train a random forest with balanced class weights
print(RandomForestClassifier(class_weight="balanced"))

# 处理方式2 进行下采样
# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]
# 读取类的大小
n_class0 = len(i_class0)
n_class1 = len(i_class1)
# For every observation of class 0, randomly sample
# 随机取样，使得class1 Observation数量和class0一样多
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)
# 连接
# 下采样
print(np.hstack((target[i_class0], target[i_class1_downsampled])))
print(np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5])

# 处理方式3 上采样
# 对于class0来说,随机创建样本直至数量和class1一样
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# 目标向量
print(np.concatenate((target[i_class0_upsampled], target[i_class1])))
# 特征矩阵
print(np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5])