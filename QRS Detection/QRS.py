# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 03:42:04 2021

@author: Kevin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy.io
from ecgdetectors import Detectors


example_dir = 'ECG.tsv'
my_dir = 'converted/vertically_segmented0_0_0.csv'
mat = 'A0001.mat'

mat_ecg = scipy.io.loadmat(mat)

_val = mat_ecg['val']
print(_val)
plt.figure()
plt.plot(_val)
plt.xlim([0, 0.1])
plt.ylim([0, 10])
plt.show()

my_ecg = np.loadtxt(my_dir)
plt.figure()
plt.plot(my_ecg)
plt.show()

unfiltered_ecg_dat = np.loadtxt(example_dir) 
unfiltered_ecg = unfiltered_ecg_dat[:, 0]
fs = 150

detectors = Detectors(fs)

# r_peaks = detectors.two_average_detector(my_ecg)
#r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
#r_peaks = detectors.swt_detector(unfiltered_ecg)
r_peaks = detectors.engzee_detector(my_ecg)
#r_peaks = detectors.christov_detector(unfiltered_ecg)
#r_peaks = detectors.hamilton_detector(unfiltered_ecg)
# r_peaks = detectors.pan_tompkins_detector(my_ecg)


plt.figure()
plt.plot(my_ecg)
plt.plot(r_peaks, my_ecg[r_peaks], 'ro')
plt.title('Detected R-peaks')
plt.xlim([0, 360])

plt.show()

tsv_r_peaks = detectors.engzee_detector(unfiltered_ecg)
plt.figure()
plt.plot(unfiltered_ecg)
plt.plot(tsv_r_peaks, unfiltered_ecg[tsv_r_peaks], 'ro')
plt.title('Detected R-peaks')
plt.xlim([0,2000])

plt.show()

print(my_ecg[r_peaks])