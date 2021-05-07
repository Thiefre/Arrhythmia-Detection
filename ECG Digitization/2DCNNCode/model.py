# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 04:55:17 2021

@author: Edgar
"""
import numpy as np
import cv2
import keras
from glob import glob
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPool2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from collections import Counter
import tensorflow as tf

filepath = input("Name the model: ")

train_path = input("Training image directory: ")
test_path = input("Testing image directory: ")

checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

batch_size = 32

IMAGE_SIZE = [128, 128]

model = Sequential()
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu', input_shape=IMAGE_SIZE+[3]))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(model.summary())

gen = ImageDataGenerator()

test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE)

train_gen = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE)

train_generator = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size)

test_generator = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size)

callbacks_list = [checkpoint]

r = model.fit(train_generator, test_data=test_generator, epochs=1)

directory = input('Images to classify: ')
images = glob(directory + '/*.png')
res = model.predict(test_generator)
reslist = []

for i in images:
    image = cv2.imread(i)
    pred = model.predict(image.reshape((1, 128, 128, 3)))
    
    y_classes = pred.argmax(axis=-1)
    reslist.append(y_classes[0])
    
result, rescount = Counter(reslist).most_common(1)[0]

print("Results:")
if(result == 0):
	print('The patient may have atrial fibrillation.')
	print('Number of irregular beats: ' + str(rescount) + '/' + str(len(images)))

elif(result == 1):
	print('The patient does not have atrial fibrillation.')
	print('Number of regular beats: ' + str(rescount) + '/' + str(len(images)))
