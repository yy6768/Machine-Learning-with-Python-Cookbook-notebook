import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate 特征矩阵
features, _ = make_blobs(n_samples=1000,
                         n_features=10,
                         centers=2,
                         cluster_std=0.5,  # 聚类的标准差
                         shuffle=True,
                         random_state=1)
# 使用 k-means 去预测分类
model = KMeans(n_clusters=2, random_state=1).fit(features)
# 获得分完类预测所得到的标签
target_predicted = model.labels_
# 使用silhouette coefficients 预测
print(silhouette_score(features, target_predicted))