# 方式1，通过条件查询直接删除它们
# Load library
import pandas as pd
# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
# 过滤
print(houses[houses['Bathrooms'] < 20])


# 方法2，定义一新特征“outliner",然后使用np.where创建条件查询

# Load library
import numpy as np
# Create feature based on boolean condition
#houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# Show data
#print(houses)

# 方法3：通过数值转换抑制某一特征异常值的影响
# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
# Show data
print(houses)