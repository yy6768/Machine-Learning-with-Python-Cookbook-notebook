import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 删除age
print(dataframe.drop('Age', axis=1).head(2))

# 删除两列
print(dataframe.drop(['Age', 'Sex'], axis=1).head(2))

# 通过列的description删除
print(dataframe.drop(dataframe.columns[1], axis=1).head(2))
