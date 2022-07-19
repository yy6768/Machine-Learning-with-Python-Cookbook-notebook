# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 更改大小
image_50x50 = cv2.resize(image, (50, 50))
# 查看新的图片
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()