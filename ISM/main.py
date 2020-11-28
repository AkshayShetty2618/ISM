import cv2
import numpy as np

img = cv2.imread('img.jpg',0)  #pass 0 to convert into gray level
ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
print(thr)
kernel = np.ones((15,15),np.uint8)

closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
blur = cv2.blur(closing,(15,15))

cv2.imshow('win1', blur)

#cv2.imshow('win1', thr)
cv2.waitKey(0)
cv2.destroyAllWindows()