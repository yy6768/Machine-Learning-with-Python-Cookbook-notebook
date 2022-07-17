# 方法1用numpy
# Load library
import numpy as np
# Create feature matrix
features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])
# Keep only observations that are not (denoted by ~) missing
print(features[~np.isnan(features).any(axis=1)])

# 方法2用pandas
import pandas as pd
# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Remove observations with missing values
print(dataframe.dropna())