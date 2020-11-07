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
    
    # Index of where to begin the cropping
    startCrop = 0
    # Index of where whitespace starts
    start = 0
    # Scans from top to bottom, the length of the image
    for i in range(0, height):
        count = i
        # If that row has whitespace, do nothing and go to next row
        if proj[i][0] <= 1500:
            pass
        # If that row is not whitespace, and there has been consecutively at least 30 whitespaces before, start the crop
        # This will only enter at the start of a signal; at the start of the next signal it will crop
        elif count-start >= 30 and proj[i][0] > 1500:
            # If the cropping image is bigger than 50 pixels in height, then crop
            if (start - startCrop >= 30):
                if((count-start)/2)+start <= height:
                        # Crops the image based on the start of the next signal, the start of the previous whitespace, and the inital cropping index
                        crop_img = img[startCrop:start, 0:width]
                        cv2.imwrite(path + '/'+ item[:len(item)-4] +'_'+str(index)+'.jpg', crop_img)
                        index += 1
            # Sets the index to start cropping
            count = i
            start = i
            startCrop = i
        # If that row is not whitespace, and there is no whitespace before
        # This will indicate the starting point of new whitespace
        else:
            start = i
    if( start - startCrop >= 30):
        crop_img = img[startCrop:start, 0:width]
        cv2.imwrite(path + '/'+ item[:len(item)-4] +'_'+str(index)+'.jpg', crop_img)
    index = 0
# Save result
# cv2.imwrite(path + '/'+ path + str(count)+'_'+str(index)+'.jpg', result)