# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:40:44 2021

@author: 20210595
"""

from sklearn.datasets import load_iris
from gama import GamaClassifier

if __name__ == '__main__':
    automl = GamaClassifier(
        max_total_time=60,
        max_eval_time=10,
        n_jobs=-1
    )
    x, y = load_iris(return_X_y=True)
    automl.fit(x, y)
    print(automl.score(x,y))