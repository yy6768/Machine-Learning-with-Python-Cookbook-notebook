# Load library
import pandas as pd
# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# 设置index
dataframe = pd.DataFrame(index=time_index)
# 新建一个特征
dataframe["Stock_Price"] = [1,2,3,4,5]
# 窗口大小为2，求出窗口时间中的平均值
print(dataframe.rolling(window=2).mean())