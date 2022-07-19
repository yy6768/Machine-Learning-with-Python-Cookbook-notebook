# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to RGB
image_bgr = cv2.imread('images/plane.jpg')
# Resize image to 50 pixels by 50 pixels
image_50x50 = cv2.resize(image_bgr, (256, 256))

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)
# 创建起始遮罩
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# 为grabCut算法使用的临时空间
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# 应用 grabCut
cv2.grabCut(image_rgb,  # 原图片
            mask,  # 初始遮罩
            rectangle,  # 定义的长方形区域
            bgdModel,  # 背景
            fgdModel,  # 背景
            5,  # Number of iterations
            cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle
# 将确定为背景的地方标记为0，否则标记为1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 把mask2减去
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# 显示
plt.imshow(image_rgb_nobg), plt.axis("off")

plt.show()

# Show mask
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()

# Show mask
plt.imshow(mask_2, cmap='gray'), plt.axis("off")
plt.show()