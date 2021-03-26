# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:32:04 2020

@author: Kevin
"""

import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
import math
import numpy as np
from scipy import stats, ndimage
import scipy.signal as signal

dest = 'converted'
try:
    files = os.listdir(dest)
except FileNotFoundError:
    os.mkdir(dest)

src = 'horizontally_segmented'
try:
    horizontal_files = os.listdir(src)
except FileNotFoundError:
    os.mkdir(dest)

def removeArtifacts( imgName ):
    img = cv2.imread(imgName)
    vals = []
    for col in range(img.shape[1]):
        currentCol = img[:, col]
        vals = np.where(currentCol < 50)
        
        if(vals[0].size != 0):
            delete = False
            pixels = set()
            oldPos = vals[0][0]
                    
            for pos in vals[0]:
                if(pos - oldPos > 25):
                    delete = True
                    break
                elif delete == False:
                    pixels.add(pos)
                    oldPos = pos
                    
            if delete == True:
                for i in pixels:
                    img[i,col] = [255,255,255]
    cv2.imwrite(imgName, img)


def convertToValues ( imgName ):
    img = cv2.imread(imgName)

    # Scans each column in the image and evaluates the value of the corresponding pixel at that column
    vals = []
    
    # img.shape[1] is width
    for col in range( img.shape[1] ):
        currentCol = img[:, col]
        signalVals = np.where(currentCol < 50)
        
        if(signalVals[0].size != 0):
            # Pixel values are not being calculated correctly. Median shows better(?) results than mean
            signalVal = np.mean(signalVals)
            vals.append(0 - signalVal )
        else:
            vals.append(None)  
            
    # plt.plot(vals)
    # plt.show()
    # return img2,vals
    print(len(vals))
    vals = [x for x in vals if x is not None]
    print("NEW VALS: ",len(vals))
    vals = ndimage.median_filter(vals, size=5)
    
    
    # Interpolates missing data
    # x = [i for i in range(len(vals)) if vals[i] == None]
    # allX = [i for i in range(len(vals)) ]
    # fp = [i for i in vals if i is not None]
    # xp = [i for i in allX if i not in x]
    # interp = np.interp(x, xp, fp)
    # vals = [i for i in vals if i is not None]
    # for i in range( len(x) ):
    #     vals.insert(x[i], interp[i])
    errorCorrection = stats.mode(vals, axis = None)
    baseCorrection = vals - errorCorrection[0]

    return baseCorrection.tolist()

for item in horizontal_files:
    # time.sleep(1)
    # print(os.path.join(src,item))
    filePath = os.path.join(src,item)
    removeArtifacts(filePath)
    _val = convertToValues(filePath)
    _val = np.array(_val)
    
    # plt.plot(_val)
    # plt.show()
    
    
    B,A = signal.butter(15, .6, output = 'ba')
    tempf = signal.filtfilt(B,A,_val)
    
    salgov = signal.savgol_filter(tempf, 7, 2)
    
    plt.plot(salgov)
    plt.show()
    
    try:
        np.savetxt(dest+'/'+ item.split('.')[0]+'.csv', salgov.reshape((1,len(salgov))), delimiter = " ", fmt = "%s")
    except:
        print("An exception occurred at: "+ dest + "/"+ item.split('.')[0]+".csv")
    