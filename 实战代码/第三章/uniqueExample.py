import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 查看所有可能的值，返回一个数组
print(dataframe['Sex'].unique())

# 显示次数
print(dataframe['Sex'].value_counts())