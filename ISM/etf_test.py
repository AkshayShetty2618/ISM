

import cv2
import numpy as np

img = cv2.imread('img.jpg',0)  #pass 0 to convert into gray level

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#Convert the image to HSV colorspace
dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) #Use gthe Hue to backproject the color histogram
ret, track_window = cv2.meanShift(dst, track_window, term_crit)