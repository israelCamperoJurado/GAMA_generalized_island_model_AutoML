# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:24:18 2021

@author: 20210595
"""

from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=1.2,
#                                                            kernel='linear',
#                                                            degree=2,
#                                                            gamma='auto',
#                                                            coef0=0.1))])
# InstanciaClase = Individual(SVC(C=1.2, kernel='linear', degree=2, gamma='auto', coef0=0.1), compile_individual)
# print(InstanciaClase)

# clf_config = {
#     RandomForestClassifier: {
#         "n_estimators": 100,
#         "criterion": "gini",
#         "max_features": 0.05,
#         "min_samples_split": 4,
#         "min_samples_leaf": 3,
#         "bootstrap": False,
#     }
# }

#InstanciaClase = Individual(clf_config, compile_individual)
#print(InstanciaClase)

## crear un main node, es un primitive node primero para preprocesamiento y luego para clasificadores

# pre = PrimitiveNode("Normalizer", "data", ['l1'])



## Primero es crear un Primitive, en Primitive.output = "data" para preprocesamiento y Primitive.output = "prediction" para las técncias de classificación

primitiveEjemplo = Primitive(['PCA.svd_solver', 'PCA.iterated_power'], "data", PCA)
# print(primitivoEjemplo.output)
# print(primitivoEjemplo.identifier)

## Ahora es crear las terminales que concuerdan con el Primitive
termiSVD_solver = Terminal("randomized", "PCA.svd_solver", "PCA.svd_solver")
termiITERATED = Terminal(9, "PCA.iterated_power", "PCA.iterated_power")
## Recuerda el orden 
listaTerminals = [termiITERATED, termiSVD_solver]

## Ahora con el primitivo vamos a crear un PrimitiveNode de pre-procesamiento

primitiveNodePreProcesamiento = PrimitiveNode(primitiveEjemplo, "data", listaTerminals)

## Con el PrimitiveNode de pre-procesamiento vamos a crear el PrimitiveNode para classificacion
## Recuerda el primer paso es crear el Primitive con Primitive.output = prediction

primitiveClassification = Primitive(['RandomForestClassifier.n_estimators', # Primitive.input
                                      'RandomForestClassifier.criterion',
                                      'RandomForestClassifier.max_features',
                                      'RandomForestClassifier.min_samples_split',
                                      'RandomForestClassifier.min_samples_leaf',
                                      'RandomForestClassifier.bootstrap'], 
                                    "prediction", # Primitive.output
                                    RandomForestClassifier # Primitive.identifier
                                    )

## Ahora es crear las terminales que concuerdan con el Primitive

terminal1 = Terminal(100, 'RandomForestClassifier.n_estimators', 'RandomForestClassifier.n_estimators')
terminal2 = Terminal("gini", 'RandomForestClassifier.criterion', 'RandomForestClassifier.criterion')
terminal3 = Terminal(0.05, 'RandomForestClassifier.max_features', 'RandomForestClassifier.max_features')
terminal4 = Terminal(3, 'RandomForestClassifier.min_samples_split', 'RandomForestClassifier.min_samples_split')
terminal5 = Terminal(1, 'RandomForestClassifier.min_samples_leaf', 'RandomForestClassifier.min_samples_leaf')
terminal6 = Terminal(False, 'RandomForestClassifier.bootstrap', 'RandomForestClassifier.bootstrap')

terminalClassification = [terminal1, terminal2, terminal3, terminal4, terminal5, terminal6]

## Ahora con el primitivo vamos a crear un PrimitiveNode de classification

## NOTA: SI NO HAY TÉCNICA DE PREPROCESAMIENTO PrimitiveNode._data_node = "data" (de tipo string), de lo contario
## es decir si hay PrimitiveNode de preprocesamiento, se tiene que tragar el PrimitiveNode de preprocesamiento, como aqui:
primitiveNodeClassification = PrimitiveNode(primitiveClassification, primitiveNodePreProcesamiento, terminalClassification)
print(primitiveNodeClassification)

#Ahora crearemos un Individual

ind = Individual(primitiveNodeClassification, compile_individual)
print(ind)

listExample = [ind for i in range(10)]