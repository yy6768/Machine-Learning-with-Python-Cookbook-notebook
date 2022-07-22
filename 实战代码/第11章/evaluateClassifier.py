# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成 features matrix and target vector
X, y = make_classification(n_samples=10000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=2,
                           random_state=1)
# 创建 logistic regression
logit = LogisticRegression()
# CV打分函数，scoring为accuracy
print(cross_val_score(logit, X, y, scoring="accuracy"))

# CV 打分函数 scoring为 precision
print(cross_val_score(logit, X, y, scoring="precision"))

# CV 打分函数 scoring为 recall
print(cross_val_score(logit, X, y, scoring="recall"))

# CV 打分函数 scoring为 f1
print(cross_val_score(logit, X, y, scoring="f1"))

# Load library
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 创建测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=1)
# 对目标向量作出预测
y_hat = logit.fit(X_train, y_train).predict(X_test)
# Calculate accuracy
print(accuracy_score(y_test, y_hat))