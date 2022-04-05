# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:20:05 2021

@author: 20210595
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
from gama.search_methods.random_search import RandomSearch
from gama.search_methods.asha import AsynchronousSuccessiveHalving
# from gama.search_methods.rf9 import RandomForestTry
from gama.search_methods.pygmo_search import SearchPygmo # gama.py ya está modificado para trabajar perron sin que lo declaremos
#from gama.search_methods.pygmo_search4 import SearchPygmo #Funcionó mas o menos
from gama.search_methods.async_ea import AsyncEA
import pandas as pd
import numpy as np
import openml


if __name__ == "__main__":

    df = pd.read_csv('classification_dataset.csv', names = ['average', 'median', 'std', 'kSQI', 'pSQI', 'skewness', 'target'])
    size=np.array(df.shape)
    X = df.iloc[:, 0:(size[1]-1)].values  
    y = df.iloc[:, (size[1]-1)].values 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )

    automl = GamaClassifier(max_total_time=180, store='models')
    # automl = GamaClassifier(
    # max_total_time=180, # in seconds
    # verbosity=logging.INFO,  # to get printed updates about search progress
    # keep_analysis_log="ecg.log",  # name for a log file to record search output in
    # )
    print("Starting `fit` which will take roughly 180 s")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print("accuracy:", accuracy_score(y_test, label_predictions))
    print("log loss:", log_loss(y_test, probability_predictions))
    
    print("automl.model", automl.model)