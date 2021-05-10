# -*- coding: utf-8 -*-
"""
Created on Fri May  7 04:34:27 2021

@author: Kevin
"""
import pandas as pd
import os
from numpy import genfromtxt
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_yaml

def prediction():
    src = 'ECG Digitization/ECG Code/converted'
    scaler = MinMaxScaler()
    
    try:
        files = os.listdir(src)
    except FileNotFoundError:
        os.mkdir(src)
    
    df = pd.DataFrame()
    
    X_data =  np.zeros((1, 12, 500))
    
    
    index = 0
    for file in files:
        data = list(genfromtxt(os.path.join(src,file), delimiter = ' '))
        
        for i in range(len(data)):
            data[i] = data[i] * 50
        resampled = signal.resample(data, 500)
        plt.plot(resampled)
        plt.show()
        X_data[0][index] = resampled
        index += 1
        
    X_data = np.swapaxes(X_data, 1 , 2)
    
    # load YAML and create model
    yaml_file = open('Network/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("Network/model.h5")
    print("Loaded model from disk")
    
    print(loaded_model.predict(X_data, verbose = 0))
    
    return loaded_model.predict(X_data, verbose = 0)