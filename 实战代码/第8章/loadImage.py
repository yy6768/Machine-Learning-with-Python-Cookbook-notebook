# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图片
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# 使用plt展示图片
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()

# 查看图像类型
print(type(image))

# 查看图像数据
print(image)

# 展示维度
print(image.shape)

# 展示第一行第一列的像素
image[0, 0]

# 加载有颜色的图像
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# 展示像素
print(image_bgr[0, 0])
# 转换成RGB格式
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# 展示图片
plt.imshow(image_rgb), plt.axis("off")
plt.show()
