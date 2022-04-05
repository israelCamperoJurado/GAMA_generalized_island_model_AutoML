# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:17:28 2021

@author: 20210595
"""

from numpy import genfromtxt
my_data = genfromtxt('x_to_save.csv', delimiter=',')
print(my_data)

for i in my_data:
    print(len(i))