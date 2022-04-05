# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:55:41 2021

@author: 20210595
"""

import os
import pickle
path_use = os.getcwd()
path = path_use.replace(os.sep, '/')
path = path + "/list_successive_halving.pkl"  
lista = [1,1,1,1,1]
with open(path, 'wb') as f:
    pickle.dump(lista, f)    
# for root, dirs, files, in os.walk(path):
#     for file in files:
#         if file.endswith(".pkl"):
#             print(file)
#             # if file == "buscar.pkl":
#             #     os.remove(file)
            
# path = path + "/"+ "list_successive_halving.pkl"
# list_successive_halving = pickle.load(open(path, "rb"))