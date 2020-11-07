# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:39:58 2020

@author: Kevin
"""

import cv2
import numpy as np
import scipy.ndimage as ndimage
import math
import os

path = 'vertically_segmented'
try:
    files = os.listdir(path)
    count = len(files)+1
except FileNotFoundError:
    os.mkdir(path)
    count = 1

index = 0
#Input image
img = cv2.imread('rotated_houghtransform.jpg')

#Thresholded image
threshimg = cv2.imread('red_channel_thresholded/red_channel_thresholded2.jpg')
height, width, channels = img.shape

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([110,50,50])
upper_range = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_range, upper_range)

mask = 255-mask

count = 0
pointset = list()
edges = cv2.Canny(mask,50,150,apertureSize = 3)
lines= cv2.HoughLines(edges, 1, np.pi/180.0, 165, np.array([]), 0, 0)

a,b,c = lines.shape
for i in range(a):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
    pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
    # cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    add = True
    for points in pointset:
        if abs(points[0] - pt1[0]) < 50:
            add = False
            break
        else:
            add = True
    if add == True:
        pointset.append(pt1)
        
pointset.sort()
oldX = 0
for points in pointset:
    crop_img = threshimg[0:points[1], oldX+1:points[0]-1]
    oldX = points[0]
    cv2.imwrite(path + '/'+ path + str(count)+'_'+str(index)+'.jpg', crop_img)
    index += 1
crop_img = threshimg[0: height, oldX+5: width]
cv2.imwrite(path + '/'+ path + str(count)+'_'+str(index)+'.jpg', crop_img)

#Writes the image to destination
# cv2.imwrite('blue.jpg', img)