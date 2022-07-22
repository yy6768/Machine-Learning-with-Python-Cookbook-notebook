# Load libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 加载手写数字数据集
digits = datasets.load_digits()
# 特征矩阵
features = digits.data
# 目标向量
target = digits.target
# 标准化
standardizer = StandardScaler()
# 逻辑回归
logit = LogisticRegression()
# 复合估计器
pipeline = make_pipeline(standardizer, logit)
# 创建KFold cv
kf = KFold(n_splits=10, shuffle=True, random_state=1)
# 执行 k-fold cross-validation
cv_results = cross_val_score(pipeline,  # Pipeline
                             features,  # Feature matrix
                             target,  # Target vector
                             cv=kf,  # Cross-validation technique
                             scoring="accuracy",  # Loss function
                             n_jobs=-1)  # Use all CPU scores
# 计算平均值
print(cv_results.mean())

# 查看结果
print(cv_results)

# Import library
from sklearn.model_selection import train_test_split
# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.1, random_state=1)
# Fit standardizer to training set
standardizer.fit(features_train)
# Apply to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

