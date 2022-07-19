# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# 转换成 RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 特征
features = []
# 计算每一个channel
colors = ("r", "g", "b")
# 生成直方图
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 原图
                             [i],  # 索引
                             None,  # 遮罩
                             [256],  # 直方图大小
                             [0, 256])  # 范围
    features.extend(histogram)
# 创建一个用于表示特征的向量
observation = np.array(features).flatten()
# 展示前五项
print(observation[0:5])

# Show RGB channel values
print(image_rgb[0, 0])

# 绘制直方图
# Import pandas
import pandas as pd

# Create some data
data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])
# Show the histogram
data.hist(grid=False)
plt.show()

# 计算每一个channel
colors = ("r", "g", "b")
# 生成直方图
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 原图
                             [i],  # 索引
                             None,  # 遮罩
                             [256],  # 直方图大小
                             [0, 256])  # 范围
    features.extend(histogram)
# 绘制
plt.plot(histogram, color=channel)
plt.xlim([0, 256])
# Show plot
plt.show()
