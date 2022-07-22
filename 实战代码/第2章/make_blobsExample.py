# load library
from sklearn.datasets import make_blobs

# load library
import matplotlib.pyplot as plt

# generate feature_matrix and target vector
features, target = make_blobs(n_samples=100,  # 样本数量
                              n_features=2,  # 特征数量
                              centers=3,  # 类别数（中心数）
                              cluster_std=0.5,  # 每个类的方差
                              shuffle=True,  # 是否洗乱数据
                              random_state=1)  # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature Matrix\n {}".format(features[:3]))
print("Target Vector\n {}".format(target[:3]))

# view scatterplot
plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()
