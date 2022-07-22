# Load library
import pandas as pd

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})
# 创建mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}
# 替换
print(dataframe["Score"].replace(scale_mapper))