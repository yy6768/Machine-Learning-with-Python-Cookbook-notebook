import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)
# 删除female的前两列
print(dataframe[dataframe['Sex'] != 'male'].head(2))