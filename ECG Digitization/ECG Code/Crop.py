# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:53:02 2020

@author: Kevin
"""
import cv2

img = cv2.imread("ecg1.png")
crop_img = img[310:1170, 65:1560]

cv2.imwrite('cropped.jpg', crop_img)
