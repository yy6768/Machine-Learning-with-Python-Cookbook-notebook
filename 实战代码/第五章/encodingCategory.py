import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([
    ["Texas"],
    ["California"],
    ["Texas"],
    ["Delaware"],
    ["Texas"]
])

# 创建 one-hot encoder
one_hot = LabelBinarizer()

# one-hot 对feature 编码
print(one_hot.fit_transform(feature))

# 输出各个类别
print(one_hot.classes_)

# 对编码结果解码
print(one_hot.inverse_transform(one_hot.transform(feature)))

# 方法2 使用pandas
# Import library
import pandas as pd

# 从feature中创建dummy编码
print(pd.get_dummies(feature[:, 0]))


multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Create 多类别编码
one_hot_multiclass = MultiLabelBinarizer()
# One-hot encode multiclass feature
print(one_hot_multiclass.fit_transform(multiclass_feature))

# 查看所有类别
print(one_hot_multiclass.classes_)