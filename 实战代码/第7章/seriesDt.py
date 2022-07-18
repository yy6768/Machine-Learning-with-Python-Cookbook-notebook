# Load library
import pandas as pd
# 空的dataFrame
dataframe = pd.DataFrame()
# 创建 150个数据
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')
print(dataframe)
# 使用dt拆解成年月日小时分钟
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute
# Show
print(dataframe.head(3))