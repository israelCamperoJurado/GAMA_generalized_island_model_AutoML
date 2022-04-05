# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:59:08 2021

@author: 20210595
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
#from gama import GamaClassifier

if __name__ == '__main__':
    #X, y = load_iris(return_X_y=True)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    variable = 0
    def create_generator():
        global variable 
        print("La variable global es", variable)
        variable += 1
        mylist = range(3)
        for i in mylist:
            yield i*i
    
    mygenerator = create_generator() # create a generator
    print(mygenerator) # mygenerator is an object!
    for i in mygenerator:
          print(i)
    
    # variable = 0
    
    # def create_generator():
    #     global variable
    #     print("El valor de la variable es", variable)
    #     variable += 1
    #     listanueva=[]
    #     mylist = range(3)
    #     for i in mylist:
    #         listanueva.append(i*i)
    #     return listanueva
    
    # mygenerator = create_generator() # create a generator
    # print(mygenerator) # mygenerator is an object!
    # for i in mygenerator:
    #      print(i)