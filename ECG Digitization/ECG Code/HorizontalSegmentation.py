# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:15:26 2020

@author: Kevin
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = 'horizontally_segmented'
try:
    files = os.listdir(path)
except FileNotFoundError:
    os.mkdir(path)

folder = 'vertically_segmented'
try:
    vertical_files = os.listdir(folder)
except FileNotFoundError:
    os.mkdir(path)
    
index = 0
for item in vertical_files:
    
    # Load as greyscale
    img = cv2.imread(os.path.join(folder,item))
    invertedimg = 255-img
    
    height, width = img.shape[:2]
    proj = np.sum(invertedimg,1)
    plt.plot(proj)
    plt.xlim([0, height])
    plt.draw()
    
    startCrop = 0
    start = 0
    for i in range(0, height):
        count = i
        # print(proj[i][0], i)
        if proj[i][0] <= 2000:
            count += 1
        elif count-start >= 40 and proj[i][0] > 2000:
            if ((count-start)/2)+start - startCrop > 40:
                if int((count-start)/2)+start <= height:
                    crop_img = img[startCrop:int((count-start)/2)+start, 0:width]
                    print(((count-start)/2)+start, startCrop, item[:len(item)-4])
                    cv2.imwrite(path + '/'+ item[:len(item)-4] +'_'+str(index)+'.jpg', crop_img)
                elif int((count-start)/2)+start > height:
                    crop_img = img[startCrop:height, 0:width]
                    cv2.imwrite(path + '/'+ item[:len(item)-4] +'_'+str(index)+'.jpg', crop_img)
            index += 1
            count = i
            start = i
            startCrop = i
        else:
            start = i
    crop_img = img[startCrop:height, 0:width]
    cv2.imwrite(path + '/'+ item[:len(item)-4] +'_'+str(index)+'.jpg', crop_img)
    index = 0
# Save result
# cv2.imwrite(path + '/'+ path + str(count)+'_'+str(index)+'.jpg', result)