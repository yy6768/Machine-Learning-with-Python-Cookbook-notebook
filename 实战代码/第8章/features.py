# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image as grayscale
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))
# Convert image data to one-dimensional vector
print(image_10x10.flatten())

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

# Load image in color
image_color = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))
# Convert image data to one-dimensional vector, show dimensions
print(image_color_10x10.flatten().shape)

# Load image in grayscale
image_256x256_gray = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_gray.flatten().shape)

# Load image in color
image_256x256_color = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_color.flatten().shape)