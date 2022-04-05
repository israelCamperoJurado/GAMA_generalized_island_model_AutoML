# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:05:55 2021

@author: 20210595
"""

# from classification import clf_config

from . import classification

clf_config = classification.clf_config
count_aux = 0
upperBound = []
lowerBound = []
vector_support = []
# otro = []
# for i in clf_config:
#     if callable(i):
#         otro.append(100) # Choose the value that you want major than 50
#         print(i, len(clf_config[i]))
#         for k in range(len(clf_config[i])):
#             otro.append(1)
# print(otro, len(otro))

count_penalty = 0
count_alpha = 0
count_dual = 0

for i in clf_config:
    if callable(i):
        # print(i)
        upperBound.append(100) # Choose the value that you want major than 50
        lowerBound.append(0)
        vector_support.append(1)
        # print(i, len(clf_config[i]))
        for h in clf_config[i].items():
            # print(h[0])
            if h[0] == 'alpha' and count_alpha == 2:
                upperBound.append(0.5)
                lowerBound.append(0) 
                vector_support.append(0)
            if h[0] == 'alpha' and (count_alpha==0 or count_alpha==1):
                upperBound.append(100.0)
                lowerBound.append(1e-3)
                vector_support.append(0)
                count_alpha += 1 
            if h[0] == 'fit_prior':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'min_samples_split':
                upperBound.append(21)
                lowerBound.append(2)
                vector_support.append(0)
            if h[0] == 'min_samples_leaf':
                upperBound.append(21)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'max_depth':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'criterion':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'max_features':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'bootstrap':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'n_estimators':
                upperBound.append(200)
                lowerBound.append(100)
                vector_support.append(0)
            if h[0] == 'learning_rate':
                upperBound.append(1.0)
                lowerBound.append(1e-3)
                vector_support.append(0)
            if h[0] == 'subsample':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'n_neighbors':
                upperBound.append(51)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'weights':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'p':
                upperBound.append(2)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'penalty' and count_penalty==1:
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'penalty' and count_penalty==0:
                upperBound.append(1)
                lowerBound.append(0)
                count_penalty += 1
                vector_support.append(0)
            if h[0] == 'loss':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'dual' and count_dual == 1:
                upperBound.append(0.4)
                lowerBound.append(0) 
                vector_support.append(0)
            if h[0] == 'dual' and count_dual == 0:
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
                count_dual += 1
            if h[0] == 'tol':
                upperBound.append(1e-1)
                lowerBound.append(1e-5)
                vector_support.append(0)
            if h[0] == 'C':
                upperBound.append(25.0)
                lowerBound.append(1e-4)
                vector_support.append(0)
            if h[0] == 'solver':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'threshold':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'linkage':
                upperBound.append(2)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'affinity':
                upperBound.append(5)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'norm':
                upperBound.append(2)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'kernel':
                upperBound.append(8)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'gamma':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'n_components':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'svd_solver':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'iterated_power':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'degree':
                upperBound.append(2.4)
                lowerBound.append(2)
                vector_support.append(0)
            if h[0] == 'include_bias':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'interaction_only':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'percentile':
                upperBound.append(100)
                lowerBound.append(1)
                vector_support.append(0)


positions = [i for i in range(len(vector_support)) if vector_support[i] == 1]
positionClassifier = [positions[i] for i in range(10)] # El díez es porque las primeras diez técnicas son de clasificacion
positionPreprocess = [positions[i] for i in range(10, len(positions))] 

# from gama.genetic_programming.components.individual import Individual
# from gama.genetic_programming.compilers.scikitlearn import compile_individual
# from gama.genetic_programming.components.primitive_node import PrimitiveNode
# from gama.genetic_programming.components.primitive import Primitive
# from gama.genetic_programming.components.terminal import Terminal
# from sklearn.naive_bayes import GaussianNB
# primitiveClassification2 = Primitive([], "prediction", GaussianNB)
# terminalClassification2 = []
# primitiveNodeClassification2 = PrimitiveNode(primitiveClassification2, "data", terminalClassification2)

# individual_play = Individual(primitiveNodeClassification2, compile_individual)
# print(individual_play)