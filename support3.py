# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:33:50 2021

@author: 20210595
"""

class Person:
    name = 'Adam'
    
p = Person()
print("Before modification", p.name)

# setting name to 'John'
setattr(p, 'name', 'John')

print('After modification: ', p.name)


#%% 
#We can set an attribute no present in the object

class Person:
    name = 'Adam'
    
p = Person()
# setting attribute name to John
setattr(p, 'name', 'John')
print('Name is:', p.name)

# setting an attribute not present in Person
setattr(p, 'age', 23)
print('Age is:', p.age)


#%%

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=1.2))])