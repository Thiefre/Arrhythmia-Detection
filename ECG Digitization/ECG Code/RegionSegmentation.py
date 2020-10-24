# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:39:58 2020

@author: Kevin
"""

import cv2
import numpy as np
import scipy.ndimage as ndimage

img = cv2.imread('rotated_houghtransform.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([110,50,50])
upper_range = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_range, upper_range)

mask = 255-mask

cv2.imshow('mask',mask)
cv2.waitKey()

edges = cv2.Canny(mask,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        # crop_img = img[0:y1, 0:x1]
        # cv2.imshow('cropped',crop_img)
        # cv2.waitKey()

cv2.imshow('img', img)
cv2.waitKey()
#Writes the image to destination
# cv2.imwrite('blue.jpg', img)