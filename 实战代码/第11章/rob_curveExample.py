# Load libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# 创建特征矩阵和目标向量
features, target = make_classification(n_samples=10000,  # 10000个样本
                                       n_features=10,  # 10个特征
                                       n_classes=2,  # 2个类别
                                       n_informative=3,  # 参与建模的特征数为3个
                                       random_state=3)  # 随机种子
# 分离训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)
# 创建LogisticRegression 分类器
logit = LogisticRegression()
# 训练模型
logit.fit(features_train, target_train)
# 预测结果
target_probabilities = logit.predict_proba(features_test)[:, 1]
# 错误和正确的比例
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
                                                               target_probabilities)
# 使用pyplot绘制ROC曲线图
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# 获取第一个observation的分类概率
print(logit.predict_proba(features_test)[0:1])

# 预测的结果
print(logit.classes_)

# 阈值大约为0.5
print("Threshold:", threshold[116])
print("True Positive Rate:", true_positive_rate[116])
print("False Positive Rate:", false_positive_rate[116])

# 阈值提升到0.8
print("Threshold:", threshold[45])
print("True Positive Rate:", true_positive_rate[45])
print("False Positive Rate:", false_positive_rate[45])

# 计算曲线面积
print(roc_auc_score(target_test, target_probabilities))