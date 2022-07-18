# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# 创建两个时间
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
# 计算持续时间
print(dataframe['Left'] - dataframe['Arrived'])

# 计算持续时间
print(pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived'])))

timedelta = pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
print(timedelta)