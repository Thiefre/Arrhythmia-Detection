# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 00:38:13 2020

@author: Kevin
"""

import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import cv2
import math
import numpy as np
from scipy import stats

dest = 'removed'
try:
    files = os.listdir(dest)
except FileNotFoundError:
    os.mkdir(dest)

src = 'horizontally_segmented'
try:
    horizontal_files = os.listdir(src)
except FileNotFoundError:
    os.mkdir(dest)


for item in horizontal_files:
    img = cv2.imread(os.path.join(src,item))
    
    arr = np.asarray(img)
    black_pixels = np.array(np.where(arr == 0))
    black_pixel_coordinates = list(zip(black_pixels[1],black_pixels[0]))
    
    black_pixel_coordinates.sort()
    
    savedNum = 0
    array = []
    pixelsOfInterest = []
    storedSet = set()
    index = 0
    array.append(black_pixel_coordinates[0])
    for item1 in black_pixel_coordinates:
        currentNum = item1[0]
        if item1[0] == array[index][0]:
            if abs(item1[1] - array[index][1]) > 50:
                storedSet.add(item1[0])
            else:
                array.append(item1)
                index += 1
        else:
            array.append(item1)
            index += 1
        

    print(storedSet)
    
    pixelMinX = 0;
    pixelMaxX = 0;
    pixelMinY = 0;
    pixelMaxY = 0;
    for item2 in black_pixel_coordinates:
        if item2[0] in storedSet:
            if item2[1] > 45:
                continue;
            else:
                if pixelMinX > item2[0]:
                    pixelMinX = item2[0]
                elif pixelMaxX < item2[0]:
                    pixelMaxX = item2[0]              
                if pixelMinY > item2[1]:
                    pixelMinY = item2[1]
                elif pixelMaxY < item2[1]:
                    pixelMaxY = item2[1]  
                pixelsOfInterest.append(item2)
    img[pixelMinY:pixelMaxY, pixelMinX:pixelMaxX] = [255,255,255]
    print(pixelsOfInterest)
    cv2.imwrite(dest + '/'+ item[:len(item)-4] +'.jpg', img)

    # plt.plot(array)
    # for x in xcoords:
    #     if(val[x]):
    #         val[x] = ycoords[x]
            