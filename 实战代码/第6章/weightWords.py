# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# 创建 the tf-idf 特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# 展示这个特征矩阵
print(feature_matrix)

# 转换为一般数组
print(feature_matrix.toarray())
# 特征名称
print(tfidf.vocabulary_)