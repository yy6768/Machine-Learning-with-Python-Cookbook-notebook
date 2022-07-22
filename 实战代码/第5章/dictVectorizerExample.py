# Import library
from sklearn.feature_extraction import DictVectorizer
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]
# 创建 dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)
# 默认是生成一个产生稀疏矩阵的dicvectorizer2
dictvectorizer2 = DictVectorizer()
# 把字典转换为特征
features = dictvectorizer.fit_transform(data_dict)
features2 = dictvectorizer2.fit_transform(data_dict)
# 查看 feature matrix
print(features)
print(features2)

# 查看特征的名字
feature_names = dictvectorizer.get_feature_names_out()
# 查看
print(feature_names)

# 还可以使用pandas
import pandas as pd
# 通过列来表明不同的特征
print(pd.DataFrame(features, columns=feature_names))