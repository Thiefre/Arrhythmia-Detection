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

def threshold():
    src = 'unthresholded_vertical'
    dest  = 'vertically_thresholded'
    
    try:
        files = os.listdir(src)
        count = len(files)+1
    except FileNotFoundError:
        os.mkdir(src)
        count = 1
    
    # cvimage = cv.imread('rotated_houghtransform.jpg')
    index = 0
    for item in files:
        src = Image.open(os.path.join(src,item))
        
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
        
        
        pil_image = image_enhanced.convert('RGB') 
        cvimage = numpy.array(pil_image) 
        cvimage = cvimage[:, :, ::-1].copy() 
        
        (b,g,r) = cv.split(cvimage)
        
        ret, thresh = cv.threshold(r, 200, 255, cv.THRESH_BINARY)
        
        thresh = cv.medianBlur(thresh, 3)
        
        cv.imwrite(dest + '/'+ str(index)+'.jpg', thresh)
        index += 1


# ret,thresh = cv.threshold(img_gray,120,255,cv.THRESH_BINARY)

# cv.imwrite('otsu_thresh.jpg', thresh)



