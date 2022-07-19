# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 增强图像
image_enhanced = cv2.equalizeHist(image)
# 显示
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

# 有色图像
image_bgr = cv2.imread("images/plane.jpg")
# 转换成  YUV 形式
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# 直方图均衡化
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
# 转换成RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
# 展示图像
plt.imshow(image_rgb), plt.axis("off")
plt.show()
