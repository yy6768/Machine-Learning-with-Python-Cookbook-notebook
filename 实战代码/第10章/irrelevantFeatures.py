# Load libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# 加载莺尾花数据
iris = load_iris()
features = iris.data
target = iris.target
# 分类数据变为数字
features = features.astype(int)
# 通过chi2来检验分类是否和这些特征有关
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 选择具有很高的f-value(ANOVA方法）
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# 选择topn的特征
from sklearn.feature_selection import SelectPercentile

# 选择前75%
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)
# 结果
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
