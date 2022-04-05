# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:55:27 2021

@author: 20210595
"""


from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

from sklearn.naive_bayes import MultinomialNB
# Pipeline: MultinomialNB(data, alpha=10.0, fit_prior=False)

primitiveClassification = Primitive(['MultinomialNB.alpha', # Primitive.input
                                      'MultinomialNB.fit_prior'],
                                    "prediction", # Primitive.output
                                    MultinomialNB # Primitive.identifier
                                    )

terminal1 = Terminal(10.0, 'alpha', 'alpha')
terminal2 = Terminal(False, 'fit_prior', 'fit_prior')

terminalClassification = [terminal1, terminal2]

primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)
#print(primitiveNodeClassification)

#Ahora crearemos un Individual

ind = Individual(primitiveNodeClassification, compile_individual)
print(ind)


from sklearn.naive_bayes import GaussianNB

primitiveClassification2 = Primitive([], "prediction", GaussianNB)
terminalClassification2 = []
primitiveNodeClassification2 = PrimitiveNode(primitiveClassification2, "data", terminalClassification2)

ind2 = Individual(primitiveNodeClassification2, compile_individual)
print(ind2)

#%%

# Pipeline: GaussianNB(FeatureAgglomeration(RobustScaler(data), FeatureAgglomeration.affinity='manhattan', FeatureAgglomeration.linkage='complete'))
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    Binarizer,
)
primitivePreprocessing1 = Primitive([],"data", RobustScaler)
terminalPreprocessing1 = []
primitiveNodePreprocessing1= PrimitiveNode(primitivePreprocessing1, "data", terminalPreprocessing1)            

primitivePreprocessing2 = Primitive(['FeatureAgglomeration.linkage',
                                    'FeatureAgglomeration.affinity'],
                                    "data", # Primitive.output
                                    FeatureAgglomeration # Primitive.identifier
                                    )
terminal12 = Terminal('complete', 'linkage', 'linkage')
terminal22 = Terminal('manhattan', 'affinity', 'affinity')
terminalPreprocessing2 = [terminal12, terminal22]
primitiveNodePreprocessing2= PrimitiveNode(primitivePreprocessing2, primitiveNodePreprocessing1, terminalPreprocessing2)            

primitiveClassification3 = Primitive([], "prediction", GaussianNB)
terminalClassification3 = []
primitiveNodeClassification3 = PrimitiveNode(primitiveClassification3, primitiveNodePreprocessing2, terminalClassification3)
ind = Individual(primitiveNodeClassification3, compile_individual)

print("segundo")
print(ind)
