# Load library
import pandas as pd
# 创建空的DataFrame
dataframe = pd.DataFrame()
# 填充数据
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
# Select 
print(dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
(dataframe['date'] <= '2002-1-1 04:00:00')])
