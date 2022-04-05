# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:36:28 2021

@author: 20210595
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
from gama.search_methods.random_search import RandomSearch
from gama.search_methods.asha import AsynchronousSuccessiveHalving
# from gama.search_methods.rf9 import RandomForestTry
from gama.search_methods.pygmo_search import SearchPygmo #NO LE MUEVAS AL try7.py con el pygmo_search23.py
#from gama.search_methods.pygmo_search4 import SearchPygmo #Funcionó mas o menos
import pandas
import openml

if __name__ == '__main__':
    #X, y = load_breast_cancer(return_X_y=True)
    datasetID = 3
    dataset = openml.datasets.get_dataset(datasetID)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute)
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    #     dataset_format="dataframe", target=dataset.default_target_attribute)
    #data = pandas.read_csv("iris.csv")
    #data.head() # to see first 5 rows
    #X = data.drop(["species"], axis = 1)
    #y = data["species"]
    #print(X)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    #El tiempo que gama tiene por default son 6 minutos
    ####automl = GamaClassifier(max_total_time=15, store="nothing", search=SearchPygmo())
    #automl = GamaClassifier(max_total_time=60000, store="nothing", search=SearchPygmo(), n_jobs=-1) # n_jobs=-1
    automl = GamaClassifier(max_total_time=60000, store="nothing", n_jobs=-1)
    #automl = GamaClassifier(max_total_time=120, store='models') 
    #automl = GamaClassifier(max_total_time=120, store="nothing", search=RandomSearch())
    # automl = GamaClassifier(max_total_time=10, store="nothing", search=RandomForestTry()) # Hemos usado rf9
    # automl = GamaClassifier(max_total_time=10, store="nothing", search=AsynchronousSuccessiveHalving())
    print("Starting `fit` which will take roughly 10 seconds.")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print('accuracy:', accuracy_score(y_test, label_predictions))
    print('log loss:', log_loss(y_test, probability_predictions))
    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
    print('log_loss', automl.score(X_test, y_test))