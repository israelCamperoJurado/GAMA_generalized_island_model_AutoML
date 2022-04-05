# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:54:35 2021

@author: 20210595
"""

import pickle 

foo = pickle.load(open("parrot.pkl", "rb"))



nueva = 2
variable = 'f%d' % nueva



variable = 'parrot%d' % nueva + ".pkl"