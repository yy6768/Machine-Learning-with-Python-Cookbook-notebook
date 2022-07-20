# Load libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 莺尾花数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
# 创建lda
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# 打印
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])

# 方差
print(lda.explained_variance_ratio_)

# 测试参数
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)
# 方差值
lda_var_ratios = lda.explained_variance_ratio_
print(lda_var_ratios)


# 计算n_components多大时才能够达到goal_var的阈值
def select_n_components(var_ratio, goal_var: float) -> int:
    # 设置初始的参数
    total_variance = 0.0
    # 初始的特征数
    n_components = 0
    # 对于每个ratio计算
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


# 运行函数
print(select_n_components(lda_var_ratios, 0.99))