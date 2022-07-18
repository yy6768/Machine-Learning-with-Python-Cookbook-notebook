# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
# 创建一个新的特征滞后一天
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)
# 展现dataframe
print(dataframe)