# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:58:33 2021

@author: 20210595
"""

# https://github.com/MIT-LCP/wfdb-python
# https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb
# https://wfdb.readthedocs.io/en/latest/

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
import shutil
import wfdb

figure(figsize=(16, 12), dpi=80)

# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
record = wfdb.rdrecord('e0112', sampfrom=1612245, sampto=1612945) 
wfdb.plot_wfdb(record=record, title='Episode of ST depression in a peak deviation of 250 microvolts', figsize=(16,10)) 
print(record.__dict__)

ann = wfdb.rdann('e0104', 'atr')
wfdb.io.show_ann_labels()

#%%

ecg_record = wfdb.rdheader('e0104')
print(ecg_record.__dict__)

#%%

signals, fields = wfdb.rdsamp('e0104')

#%%

ann = wfdb.rdann('e0104', 'atr')
print(ann.__dict__)

#%%

wfdb.io.show_ann_labels()

#%%

dbs = wfdb.get_dbs()

#%%

list_recording = wfdb.get_record_list('stdb')

#%%

wfdb.io.show_ann_classes()

#%%

# Demo 2 - Read certain channels and sections of the WFDB record using the simplified 'rdsamp' function
# which returns a numpy array and a dictionary. Show the data.
signals, fields = wfdb.rdsamp('e0103', sampfrom=100, sampto=15000)
print(signals)
print(fields)

# # Can also read the same files hosted on Physionet
# signals2, fields2 = wfdb.rdsamp('e0103', channels=[14, 0, 5, 10], sampfrom=100, sampto=15000, pn_dir='ptbdb/patient001/')