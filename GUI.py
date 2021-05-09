# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:45:07 2021

@author: Kevin
"""

from tkinter import *   
from tkinter import filedialog
import cv2
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageTk
import numpy
import os
import HorizontalSegmentation
import Predict
import Converted
  

global isClicked

def threshold(thresh_level, image, index):
    dest  = 'ECG Digitization/ECG Code/vertically_thresholded'
    src = image
    
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
    
    (b,g,r) = cv2.split(cvimage)
    
    ret, thresh = cv2.threshold(r, thresh_level, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.medianBlur(thresh, 3)
    
    cv2.imwrite(dest + '/'+ str(index)+'.jpg', thresh)
    return thresh

def select_img():
    isClicked = True
    file_path = filedialog.askopenfilename()
    print(file_path)
    global refPt, endPt, ind
    refPt = []
    endPt = []
    cropping = False
    ind = 0
    def click_and_crop(event, x, y, flags, param):
    	# grab references to the global variables
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		refPt.append((x, y))
    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the cropping operation is finished
            endPt.append((x, y))
    		# draw a rectangle around the region of interest
            for pt,pt1 in zip(refPt, endPt):
                cv2.rectangle(image, pt, pt1, (0, 255, 0), 2)
            cv2.imshow("image", image)
            
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(file_path)
    image_crop = cv2.imread(file_path)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:

        ind +=1
    	# display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
    	# if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt.clear();
            endPt.clear();
        elif key == ord("q"):
            break
        elif key == ord("c"):
            index = 0

            for pt,pt1 in zip(refPt, endPt):
                cv2.imwrite('ECG Digitization/ECG Code/unthresholded_vertical/' + str(index)+'.jpg', image_crop[pt[1]:pt1[1], pt[0]:pt1[0]])
                index += 1
    # if there are two reference points, then crop the region of interest
    # from teh image and display it

    # close all open windows
    displayImages(50)
    cv2.destroyAllWindows()

def displayImages(val):
    for panel in panelMemory:
        panel.destroy()
    panelMemory.clear()
    src = 'ECG Digitization/ECG Code/unthresholded_vertical'
    files = os.listdir(src)
    index = 0
    for file in files:
        img = cv2.imread(os.path.join(src,file))
        im = Image.fromarray(img)
        thresh_image = threshold(int(val), im, index)
        thresh_im = Image.fromarray(thresh_image)
        photo = ImageTk.PhotoImage(thresh_im)
        panel = Label(image = photo)
        panel.image = photo
        panel.pack(side = "left", padx = 20, pady = 20)
        panelMemory.append(panel)
        index += 1

def predict():
    for label in labelMemory:
        label.destroy()
    labelMemory.clear()
    HorizontalSegmentation.horizontal_segment()
    convert = Converted.convert_image()
    pred = Predict.prediction()
    if round(pred[0][0], 0) == 1:
        y = "Atrial Flutter/Fibrillation"
    else:
        y = "Other"
    text = Label(text = y)
    text.pack(side = "bottom")
    labelMemory.append(text)
    
isClicked = False
panelMemory = []
labelMemory = []
window = Tk()  
window.geometry("1280x720")  
window.title('Arrhythmia Detection App')

file_path = ''

select = Button(window, text = "Select Image", command = select_img)
select.pack()  

predict_button = Button(window, text = "Predict", command = predict)
predict_button.pack()

threshold_scale = Scale(window, orient=HORIZONTAL, from_=0, to=255, length=255, label="Threshold Level", command = displayImages)
threshold_scale.pack(anchor=CENTER)

window.mainloop()  