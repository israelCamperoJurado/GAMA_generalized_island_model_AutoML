# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:33:05 2021

@author: 20210595
"""
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class ScoreAmot(object):
    _count = 0
    def __init__(self, x_train, y_train, pipeline):
        self.X = x_train
        self.y = y_train
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=1, stratify=self.y)
        self.pipeline = pipeline 
        ScoreAmot._count += 1
        self.id = ScoreAmot._count
        
    def evaluation(self):
        clf = self.pipeline 
        print("Classifier, assessing a vector for the creation of a pipeline", self.id)
        try:
            clf.fit(self.x_train, self.y_train)
            score_val = clf.score(self.x_train, self.y_train)
            print('Vector valid... Generating pipeline')
        except: 
            #print("Ganancia a -100") 
            print("Time out for this model")
            score_val = -100 
        return score_val

