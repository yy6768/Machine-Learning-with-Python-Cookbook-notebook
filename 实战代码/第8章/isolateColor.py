# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image
image_bgr = cv2.imread('images/plane.jpg')
# 转化 BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# 定义两种蓝色 in HSV
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])
# 生成掩码
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# 进行掩码操作
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
# 转换 BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# 显示 image
plt.imshow(image_rgb), plt.axis("off")
plt.show()