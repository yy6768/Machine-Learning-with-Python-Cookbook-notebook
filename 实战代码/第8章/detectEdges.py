# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_gray = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 计算中值强度
median_intensity = np.median(image_gray)
# 设置阈值
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# 运用 canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
# 显示
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()
