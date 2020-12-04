'''HSV Segmentaion'''

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from matplotlib.colors import hsv_to_rgb

img1 = cv2.imread('D:\ISM\python code\image1_after_ACE.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#plt.imshow(img1)
#plt.show()
hsv_img = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_img)
plt.show()       

lower = np.array([7, 84, 125])
upper = np.array([168, 174, 92])

lo_square = np.full((10, 10, 3), lower, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), upper, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

mask = cv2.inRange(hsv_img, lower, upper)

result = cv2.bitwise_and(img1, img1, mask=mask)
plt.figure(figsize=(15,15))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="viridis")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()