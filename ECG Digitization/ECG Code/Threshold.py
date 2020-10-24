# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 00:32:44 2020

@author: Kevin
"""
import cv2 as cv
from PIL import Image
from PIL import ImageEnhance
import numpy
import os

path = 'remove_all_red'
try:
    files = os.listdir(path)
    count = len(files)+1
except FileNotFoundError:
    os.mkdir(path)
    count = 1

# cvimage = cv.imread('rotated_houghtransform.jpg')

src = Image.open('rotated_houghtransform.jpg')

enh_bri = ImageEnhance.Brightness(src)

brightness = 1
image_brightened = enh_bri.enhance(brightness)

enh_col = ImageEnhance.Color(image_brightened)
color = 1.5
image_colored = enh_col.enhance(color)

enh_con = ImageEnhance.Contrast(image_colored)
contrast = 1.1
image_contrasted = enh_con.enhance(contrast)

enh_sha = ImageEnhance.Sharpness(image_contrasted)
sharpness = 1.3
image_enhanced = enh_sha.enhance(sharpness)

image_enhanced.show()

pil_image = image_enhanced.convert('RGB') 
cvimage = numpy.array(pil_image) 
cvimage = cvimage[:, :, ::-1].copy() 

bilateral = cv.bilateralFilter(cvimage, 10, 10, 100) 

# hsv = cv.cvtColor(bilateral, cv.COLOR_BGR2HSV)
# mask1 = cv.inRange(hsv, (0, 10, 20), (15, 255, 255))
# mask2 = cv.inRange(hsv, (165, 10, 20), (180, 255, 255))
# mask = cv.bitwise_or(mask1,mask2)

(b,g,r) = cv.split(cvimage)

ret,redThresh = cv.threshold(r, 80, 255, cv.THRESH_BINARY)

cv.imwrite(path + '/'+ path + str(count)+'.jpg', redThresh)


# ret,thresh = cv.threshold(img_gray,120,255,cv.THRESH_BINARY)

# cv.imwrite('otsu_thresh.jpg', thresh)



