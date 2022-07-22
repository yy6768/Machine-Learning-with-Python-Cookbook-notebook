# Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# 莺尾花数据集
iris = datasets.load_iris()
# 特征矩阵
features = iris.data
# 特征向量
target = iris.target
# 创建一个数组包含所有类别的名字
class_names = iris.target_names
# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)
# sigmoid函数回归
classifier = LogisticRegression()
# 训练并且预测
target_predicted = classifier.fit(features_train,
                                  target_train).predict(features_test)
# confusion_matrix创建混淆矩阵
matrix = confusion_matrix(target_test, target_predicted)
# 创建 pandas dataframe（绘制需要是pandas的格式）
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# 创建heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
