import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean
# of each group
print(dataframe.groupby('Sex').mean())

# 对某列计数
print(dataframe.groupby('Survived')['Name'].count())
# 对某列求平均值
print(dataframe.groupby(['Sex','Survived'])['Age'].mean())