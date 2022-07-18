# Load libraries
import pandas as pd
import numpy as np

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# set index
dataframe = pd.DataFrame(index=time_index)
# 创建一组有缺失的数据
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]
# 插入缺失值
print(dataframe.interpolate())

# 前值填充
print(dataframe.ffill())

# 后值填充
print(dataframe.bfill())

# 非线性
print(dataframe.interpolate(method="quadratic"))

# 限制
print(dataframe.interpolate(limit=1, limit_direction="forward"))