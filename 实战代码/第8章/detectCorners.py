# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_bgr = cv2.imread("images/plane.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)
# 设置 corner detector 参数
block_size = 2
aperture = 29
free_parameter = 0.04
# 搜索corner
detector_responses = cv2.cornerHarris(image_gray,  # 原图
                                      block_size,  # 每个像素周围的邻居大小
                                      aperture,  # 使用的Sobel核大小
                                      free_parameter)  # 自由参数，越大可以识别越软的corner
# 将探测后的结果存储
detector_responses = cv2.dilate(detector_responses, None)
# 只要探测到的值大于阈值（这里是0.02的比例），设置成黑色
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255, 255, 255]
# 转换成灰度图像
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# 显示
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

# 显示可能的 corners
plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
plt.show()

#  使用goodFeaturesToTrack
# Load images
image_bgr = cv2.imread('images/plane.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# Number of corners to detect
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25
# 检测
corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,  # 角点个数
                                  minimum_quality_score,  # 最低阈值
                                  minimum_distance)  # 最短的距离
corners = np.float32(corners)
# 圈出每个角点
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (int(x), int(y)), 10, (255, 255, 255))
# Convert to grayscale
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# Show image
plt.imshow(image_rgb, cmap='gray'), plt.axis("off")
plt.show()
