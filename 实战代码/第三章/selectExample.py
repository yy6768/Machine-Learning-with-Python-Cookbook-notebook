# 引入库
import pandas as pd

url = 'titanic.csv'

dataframe = pd.read_csv(url)

# 单条件查询

print(dataframe[dataframe['Sex'] == 'female'].head(2))

# 多条件查询
print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 50)])