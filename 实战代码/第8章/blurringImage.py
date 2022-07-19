# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 模糊
image_blurry = cv2.blur(image, (5,5))
# 展示
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

# 以100*100为区域均值模糊图像
image_very_blurry = cv2.blur(image, (100,100))
# Show image
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

# 创造 kernel
kernel = np.ones((5,5)) / 25.0
# 展示 kernel
print(kernel)

# 运用卷积核
image_kernel = cv2.filter2D(image, -1, kernel)
# Show image
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()