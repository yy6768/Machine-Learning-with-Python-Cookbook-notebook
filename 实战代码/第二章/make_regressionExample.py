# load library
from sklearn.datasets import make_regression

# 生成 features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples=100,  # 样本数量
                                                 n_features=3,  # 特征
                                                 n_informative=3,  # 参与建模的特征数
                                                 n_targets=1,   # 因变量个数
                                                 noise=0.0,     # 噪声
                                                 coef=True,     # 是否输出coef标志
                                                 random_state=1)    # 固定值表示每次调用参数一样的数据

# view feature matrix and target vector
print("Feature Matrix \n {}".format(features[:3]))
print("Target Vector \n {}".format(target[:3]))
