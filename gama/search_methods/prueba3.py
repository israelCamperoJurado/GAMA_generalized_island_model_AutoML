# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:11:06 2021

@author: 20210595
"""

class Power():
    nueva =[]
    def __init__(self):
        self.algo = 0
        
    def fitness(self, x):
        Power.nueva.append(1)
        return x
    
insta1 = Power()
insta1.fitness(2)

insta2 = Power()
insta2.fitness(2)