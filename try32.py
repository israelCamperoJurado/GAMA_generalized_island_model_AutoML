# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:01:01 2021

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
# from multiprocessing import freeze_support
import pandas
#import openml
    
if __name__ == '__main__':
    # freeze_support()
    X, y = load_breast_cancer(return_X_y=True)
    # dataset = openml.datasets.get_dataset(1111)
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    # dataset_format="dataframe", target=dataset.default_target_attribute
    # )

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    automl = GamaClassifier(max_total_time=180, store="all", n_jobs=-1)

    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print('accuracy:', accuracy_score(y_test, label_predictions))
    print('log loss:', log_loss(y_test, probability_predictions))
    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
    print('log_loss', automl.score(X_test, y_test))