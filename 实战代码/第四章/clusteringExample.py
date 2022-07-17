import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

df = pd.DataFrame(features, columns=["feature_1", "feature_2"])

#  k-means 聚类
clusterer = KMeans(3, random_state=0)

# 过滤数据
clusterer.fit(features)

# 预测分类
df['group'] = clusterer.predict(features)

print(df.head(5))