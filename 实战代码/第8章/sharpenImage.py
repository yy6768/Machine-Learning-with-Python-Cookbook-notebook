# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 创建卷积核
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
# 锐化
image_sharp = cv2.filter2D(image, -1, kernel)
# 显示图片
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()
