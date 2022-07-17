# load scikit-learn's datasets
from sklearn import datasets

# 加载 digits 数据集
digits = datasets.load_digits()

# 创建 features matrix
features = digits.data
print(features)
# 创建 target vector
target = digits.target
print(target)
#  查看第一个 observation
print(features[0])