# Load libraries
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# 生成特征矩阵和目标向量
features, target = make_regression(n_samples=100,
                                   n_features=3,
                                   random_state=1)
# 生成测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.10, random_state=1)


# 创建一个自定义的评估度量函数
def custom_metric(target_test, target_predicted):
    # 计算r2_score
    r2 = r2_score(target_test, target_predicted)
    # 返回
    return r2


# 定义scorer，然后greater_is_better表明分数越高模型越好
score = make_scorer(custom_metric, greater_is_better=True)
# 脊回归
classifier = Ridge()
# 训练脊回归模型
model = classifier.fit(features_train, target_train)
# 使用自定义的scorer
print(score(model, features_test, target_test))

# 预测值
target_predicted = model.predict(features_test)
# 使用r2_score评分
print(r2_score(target_test, target_predicted))
