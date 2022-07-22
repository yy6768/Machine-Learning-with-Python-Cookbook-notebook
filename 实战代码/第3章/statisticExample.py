import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)
# 计算统计属性值
# print('Maximum:', dataframe['Age'].max())
# print('Minimum:', dataframe['Age'].min())
# print('Mean:', dataframe['Age'].mean())
# print('Sum:', dataframe['Age'].sum())
# print('Count:', dataframe['Age'].count())
# 计算所有属性的出现次数
print(dataframe.count())