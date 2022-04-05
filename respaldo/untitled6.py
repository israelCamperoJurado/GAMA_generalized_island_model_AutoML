# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 13:22:37 2021

@author: 20210595
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

import pandas

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    
    data_storage = {}
    for percen in [0.9, 0.5, 0.3]:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=percen, stratify=y, random_state=0)
        data_storage["X_train"+str(percen)] = X_train
        data_storage["y_train"+str(percen)] = y_train