# Load library
import pandas as pd
# Create DataFrame
data_a = {'id': ['1', '2', '3'],
'first': ['Alex', 'Amy', 'Allen'],
'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])
# Create DataFrame
data_b = {'id': ['4', '5', '6'],
'first': ['Billy', 'Brian', 'Bran'],
'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])
# 连接行
# print(pd.concat([dataframe_a, dataframe_b], axis=0))
print(pd.concat([dataframe_a, dataframe_b], axis=1))