# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 加载 image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# 保存
print(cv2.imwrite("images/plane_new.jpg", image))