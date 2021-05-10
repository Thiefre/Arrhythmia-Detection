# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 01:54:50 2021

@author: Kevin
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import scipy.io as scipy
from scipy import signal
from sklearn.model_selection import train_test_split
from keras.models import model_from_yaml
from ecgdetectors import Detectors

data_paths = ['China-PhysioNet/', 'WFDB/']
SNOMED_codes = pd.read_csv('https://raw.githubusercontent.com/physionetchallenges/physionetchallenges.github.io/master/2020/Dx_map.csv')
atrial_fibrillation = '164889003'
atrial_fib_and_flutter = '195080001'


# fs = 150

# detectors = Detectors(fs)



df = pd.DataFrame()
classes = []

X_all = np.zeros((13797, 300, 12))
index = 0
total_peak = 0

for datapath in data_paths:
    files = os.listdir(datapath)
    for file in files:
        if file.endswith('.mat'):
            mat = scipy.loadmat(datapath+file)
            
            arr = list(mat.values())
            arr = arr[0]
            resampled = signal.resample(arr[0][0:1500], 300)
            # plt.plot(resampled)
            # plt.show()
    
            # arr = pad_sequences(arr, maxlen = 1000, truncating = 'post', padding = 'post')
            # r_peaks = detectors.engzee_detector(resampled)

            # rstd = np.var(r_peaks)
            # resampled = np.append(resampled, rstd)
            
            swapped = np.swapaxes([resampled], 0 , 1)
            

            X_all[index] = swapped
            index += 1
            
            df_sub = pd.DataFrame({'I':[arr[0]], 'II':[arr[1]], 'III':[arr[2]], 'aVR':[arr[3]],
                                   'aVL':[arr[4]], 'aVF':[arr[5]], 'V1':[arr[6]], 'V2':[arr[7]], 'V3':[arr[8]],
                                   'V4':[arr[9]], 'V5':[arr[10]], 'V6':[arr[11]]})
            df = pd.concat([df, df_sub],axis = 0)
            
            
        if file.endswith('.hea'):
            input_header_file = datapath+file
            with open(input_header_file,'r') as f:
                header_data=f.readlines()
                dx = header_data[15][5:].strip('\n').split(',')
                dx = int(dx[0])
                if dx == int(atrial_fibrillation) or dx == int(atrial_fib_and_flutter):
                    classes.append(1)
                else:
                    classes.append(0)
                    
df.index = range(0,len(df))
df['Output'] = classes
classes = np.array(classes)

X_train, X_test,y_train,y_test=train_test_split(X_all, classes)

# Simple CNN for ECG data
# Simple CNN for ECG data
model = Sequential()
model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (300,12)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
# compile the model - use categorical crossentropy, and the adam optimizer
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 1, batch_size = 32, verbose = 0)
yhat = model.predict(X_test, verbose = 0)

total_correct = 0
for i in range(len(yhat)):
    if yhat[i] == y_test[i] and yhat[i] == 1:
        total_correct += 1

print(total_correct/len(yhat))


model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("model.h5")
print("Saved model to disk")






