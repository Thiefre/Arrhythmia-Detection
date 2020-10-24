# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:39:58 2020

@author: Kevin
"""

import cv2
import numpy as np
import scipy.ndimage as ndimage

img = cv2.imread('cropped.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,3,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
#Rotates image to make the detected line horizontal

img_rotated = ndimage.rotate(img, 180*theta/3.1415926-(90), cval = 255)

#Writes the image to destination
cv2.imwrite('rotated_houghtransform.jpg',img_rotated)