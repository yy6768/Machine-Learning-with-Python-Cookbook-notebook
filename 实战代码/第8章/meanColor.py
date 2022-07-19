# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 加载
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# 计算每个channel的平均值
channels = cv2.mean(image_bgr)
# 交换blue和red的值 (making it RGB, not BGR)
observation = np.array([(channels[2], channels[1], channels[0])])
# 展示 mean channel values
print(observation)

# 显示
plt.imshow(observation), plt.axis("off")
plt.show()
