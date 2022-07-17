# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Load library
import pandas as pd

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])


# 创建一个函数
def add_ten(x):
    return x + 10


# Create transformer
ten_transformer = FunctionTransformer(add_ten)
# 运用transformer
print(ten_transformer.transform(features))


# 将features创建为DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# 使用第三章的apply函数
print(df.apply(add_ten))