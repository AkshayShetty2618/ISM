
import cv2
import numpy as np




image = cv2.imread("img.jpg")
image_resize = cv2.resize(image,(1024,1024))

grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)

kernel = cv2.getStructuringElement(1,(17,17))
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
final_image = cv2.inpaint(image_resize,threshold,1,cv2.INPAINT_TELEA)
cv2.imshow("Display window", final_image)

cv2.waitKey(0)
