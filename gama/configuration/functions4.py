# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:26:39 2021

@author: 20210595
"""

def _boo_to_int(value):
    return 0.9 if value == True else 0.1


def _string_to_int(value, **kwargs):
    for element in kwargs:
        if element == value:
            if kwargs[element]==0:
                kwargs[element] = 0.1
            else:
                kwargs[element] = kwargs[element] - 0.1
            return kwargs[element] 

print(_boo_to_int(False))

print(_string_to_int('l2', l1=0, l2=1, l3=2))