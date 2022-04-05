# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:41:45 2021

@author: 20210595
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:23:56 2021

@author: 20210595
"""
# nueva = 2
# variable = 'f%d' % nueva
# with open('output.txt', 'w') as variable:
#     print(variable)
#     variable.write('Hi there!')

# with open('output.txt', 'w') as f:
#     f.write('Hi there!')
#%%

# from gama.configuration.bounds_pygmo import (
#     upperBound, 
#     lowerBound, 
#     vector_support
#     ) 

# from bounds_pygmo import (
#     upperBound, 
#     lowerBound, 
#     vector_support
#     ) 
import random 

from space_to_vector2 import export, positions, positionClassifier, positionPreprocess, upperBound, lowerBound, vector_support
from typing import Any
from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal


import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    f_classif,
    VarianceThreshold,
)

class ValuesSearchSpace(object):
    def __init__(self, x:Any):
        self.x = x
        
    def get_individuals(self):
        dictionary_pygmo = {}
        dictionary_pygmo.update({'GaussianNB': self.x[0]})
        dictionary_pygmo.update({'BernoulliNB': self.x[1]})
        dictionary_pygmo.update({'BernoulliNB.alpha': self.x[2]})
        dictionary_pygmo.update({'BernoulliNB.fit_prior': self._int_to_bool(round(self.x[3]))})
        dictionary_pygmo.update({'MultinomialNB': self.x[4]})
        dictionary_pygmo.update({'MultinomialNB.alpha': self.x[5]})
        dictionary_pygmo.update({'MultinomialNB.fit_prior': self._int_to_bool(round(self.x[6]))})
        dictionary_pygmo.update({'DecisionTreeClassifier': self.x[7]})
        dictionary_pygmo.update({'DecisionTreeClassifier.criterion': self._int_to_string(round(self.x[8]), gini=0, entropy=1)})
        dictionary_pygmo.update({'DecisionTreeClassifier.max_depth': round(self.x[9])})
        dictionary_pygmo.update({'DecisionTreeClassifier.min_samples_split': round(self.x[10])})
        dictionary_pygmo.update({'DecisionTreeClassifier.min_samples_leaf': round(self.x[11])})
        dictionary_pygmo.update({'ExtraTreesClassifier': self.x[12]})
        dictionary_pygmo.update({'ExtraTreesClassifier.n_estimators': round(self.x[13])})
        dictionary_pygmo.update({'ExtraTreesClassifier.criterion': self._int_to_string(round(self.x[14]), gini=0, entropy=1)})
        dictionary_pygmo.update({'ExtraTreesClassifier.max_features': self.x[15]})
        dictionary_pygmo.update({'ExtraTreesClassifier.min_samples_split': round(self.x[16])})
        dictionary_pygmo.update({'ExtraTreesClassifier.min_samples_leaf': round(self.x[17])})
        dictionary_pygmo.update({'ExtraTreesClassifier.bootstrap': self._int_to_bool(round(self.x[18]))})
        dictionary_pygmo.update({'RandomForestClassifier': self.x[19]})
        dictionary_pygmo.update({'RandomForestClassifier.n_estimators': round(self.x[20])})
        dictionary_pygmo.update({'RandomForestClassifier.criterion': self._int_to_string(round(self.x[21]), gini=0, entropy=1)})
        dictionary_pygmo.update({'RandomForestClassifier.max_features': self.x[22]})
        dictionary_pygmo.update({'RandomForestClassifier.min_samples_split': round(self.x[23])})
        dictionary_pygmo.update({'RandomForestClassifier.min_samples_leaf': round(self.x[24])})
        dictionary_pygmo.update({'RandomForestClassifier.bootstrap': self._int_to_bool(round(self.x[25]))})
        dictionary_pygmo.update({'GradientBoostingClassifier': self.x[26]})
        dictionary_pygmo.update({'GradientBoostingClassifier.n_estimators': round(self.x[27])})
        dictionary_pygmo.update({'GradientBoostingClassifier.learning_rate': self.x[28]})
        dictionary_pygmo.update({'GradientBoostingClassifier.max_depth': round(self.x[29])})
        dictionary_pygmo.update({'GradientBoostingClassifier.min_samples_split': round(self.x[30])})
        dictionary_pygmo.update({'GradientBoostingClassifier.min_samples_leaf': round(self.x[31])})
        dictionary_pygmo.update({'GradientBoostingClassifier.subsample': self.x[32]})
        dictionary_pygmo.update({'GradientBoostingClassifier.max_features': self.x[33]})
        dictionary_pygmo.update({'KNeighborsClassifier': self.x[34]})
        dictionary_pygmo.update({'KNeighborsClassifier.n_neighbors': round(self.x[35])})
        dictionary_pygmo.update({'KNeighborsClassifier.weights': self._int_to_string(round(self.x[36]), uniform=0, distance=1)})
        dictionary_pygmo.update({'KNeighborsClassifier.p': round(self.x[37])})
        dictionary_pygmo.update({'LinearSVC': self.x[38]})
        dictionary_pygmo.update({'LinearSVC.penalty': self._int_to_string(round(self.x[39]), l1=0, l2=1)})
        dictionary_pygmo.update({'LinearSVC.loss': self._int_to_string(round(self.x[40]), hinge=0, squared_hinge=1)})
        dictionary_pygmo.update({'LinearSVC.dual': self._int_to_bool(round(self.x[41]))})
        dictionary_pygmo.update({'LinearSVC.tol': self.x[42]})
        dictionary_pygmo.update({'LinearSVC.C': self.x[43]})
        if dictionary_pygmo['LinearSVC.penalty'] == 'l1':
            dictionary_pygmo['LinearSVC.loss'] = 'squared_hinge'
        if (dictionary_pygmo['LinearSVC.penalty'] == 'l2') and (dictionary_pygmo['LinearSVC.loss'] == 'hinge') and (dictionary_pygmo['LinearSVC.dual'] == False):
            dictionary_pygmo['LinearSVC.dual'] == True
        if (dictionary_pygmo['LinearSVC.penalty'] == 'l1') and (dictionary_pygmo['LinearSVC.loss'] == 'squared_hinge') and (dictionary_pygmo['LinearSVC.dual'] == True):
            dictionary_pygmo['LinearSVC.dual'] == False
        dictionary_pygmo.update({'LogisticRegression': self.x[44]})
        dictionary_pygmo.update({'LogisticRegression.penalty': self._int_to_string(round(self.x[45]), l2=0)})
        dictionary_pygmo.update({'LogisticRegression.C': self.x[46]})
        dictionary_pygmo.update({'LogisticRegression.dual': self._int_to_bool(round(self.x[47]))})
        dictionary_pygmo.update({'LogisticRegression.solver': self._int_to_string(round(self.x[48]), lbfgs=0)})
        dictionary_pygmo.update({'Binarizer': self.x[49]})
        dictionary_pygmo.update({'Binarizer.threshold': self.x[50]})
        dictionary_pygmo.update({'FastICA': self.x[51]})
        dictionary_pygmo.update({'FastICA.tol': self.x[52]})
        dictionary_pygmo.update({'FeatureAgglomeration': self.x[53]})
        dictionary_pygmo.update({'FeatureAgglomeration.linkage': self._int_to_string(round(self.x[54]), ward=0, complete=1, average=2)})
        dictionary_pygmo.update({'FeatureAgglomeration.affinity': self._int_to_string(round(self.x[55]), euclidean=0, l1=1, l2=2, manhattan=3, cosine=4, precomputed=5)})
        if dictionary_pygmo['FeatureAgglomeration.linkage'] == 'ward':
            dictionary_pygmo['FeatureAgglomeration.affinity'] = 'euclidean'
        dictionary_pygmo.update({'MaxAbsScaler': self.x[56]})
        dictionary_pygmo.update({'MinMaxScaler': self.x[57]})
        dictionary_pygmo.update({'Normalizer': self.x[58]})
        dictionary_pygmo.update({'Normalizer.norm': self._int_to_string(round(self.x[59]), l1=0, l2=1, max=2)})
        dictionary_pygmo.update({'Nystroem': self.x[60]})
        dictionary_pygmo.update({'Nystroem.kernel': self._int_to_string(round(self.x[61]), rbf=0, cosine=1, chi2=2, laplacian=3, polynomial=4, poly=5, linear=6, additive_chi2=7, sigmoid=8)})
        dictionary_pygmo.update({'Nystroem.gamma': self.x[62]})
        dictionary_pygmo.update({'Nystroem.n_components': round(self.x[63])})
        dictionary_pygmo.update({'PCA': self.x[64]})
        dictionary_pygmo.update({'PCA.svd_solver': self._int_to_string(round(self.x[65]), randomized=0)})
        dictionary_pygmo.update({'PCA.iterated_power': round(self.x[66])})
        dictionary_pygmo.update({'PolynomialFeatures': self.x[67]})
        dictionary_pygmo.update({'PolynomialFeatures.degree': round(self.x[68])})
        dictionary_pygmo.update({'PolynomialFeatures.include_bias': self._int_to_bool(round(self.x[69]))})
        dictionary_pygmo.update({'PolynomialFeatures.interaction_only': self._int_to_bool(round(self.x[70]))})
        dictionary_pygmo.update({'RBFSampler': self.x[71]})
        dictionary_pygmo.update({'RBFSampler.gamma': self.x[72]})
        dictionary_pygmo.update({'RobustScaler': self.x[73]})
        dictionary_pygmo.update({'StandardScaler': self.x[74]})
        dictionary_pygmo.update({'SelectFwe': self.x[75]})
        dictionary_pygmo.update({'SelectFwe.alpha': self.x[76]})
        #dictionary_pygmo.update({'SelectFwe.score_func': {f_classif: None}})
        dictionary_pygmo.update({'SelectFwe.score_func': f_classif})
        dictionary_pygmo.update({'SelectPercentile': self.x[77]})
        dictionary_pygmo.update({'SelectPercentile.percentile': round(self.x[78])})
        #dictionary_pygmo.update({'SelectPercentile.score_func': {f_classif: None}})
        dictionary_pygmo.update({'SelectPercentile.score_func': f_classif})
        dictionary_pygmo.update({'VarianceThreshold': self.x[79]})
        dictionary_pygmo.update({'VarianceThreshold.threshold': self.x[80]})
        
        new_individuals = self._index_function(x=self.x, dictionary_pos=dictionary_pygmo)
        # individual_p = self._create_individual(dictionary_values=dictionary_pygmo, position=newpositions[0])
        # return new_individuals, positions_to_send
        return new_individuals
    
    def _int_to_string(self, value, **kwargs):
        for element in kwargs:
            if kwargs[element] == value:
                return element 
            
    def _int_to_bool(self, value):
        return True if value == 1 else False
    
    def _index_function(self, x, dictionary_pos):            
        list_index_techniques_to_use_before = [i for i in positionPreprocess if x[i] > 90]
        valuesPreprocess = [x[i] for i in list_index_techniques_to_use_before]
        valuesPreprocess.sort(reverse=True)
        list_index_techniques_to_use = []
        for i in valuesPreprocess:
            for j in range(len(x)):
                if x[j] == i:
                    list_index_techniques_to_use.append(j)
        valueIndicesClassifiers = [x[i] for i in positionClassifier]
        max_value = max(valueIndicesClassifiers)
        max_index = valueIndicesClassifiers.index(max_value)
        indexClassifier = positionClassifier[max_index]
        #The last index is the classifier
        list_index_techniques_to_use.append(indexClassifier)
        contador = 0
        for i in list_index_techniques_to_use:
            if contador == 0:
                data_node = "data"
            primitive_node_final = self._create_individual(dictionary_pos, i, data_node)
            data_node = primitive_node_final
            contador += 1
            if len(list_index_techniques_to_use) == contador:
                ind = Individual(primitive_node_final, compile_individual)
        #return ind, list_index_techniques_to_use 
        return ind
 
    def _create_individual(self, dictionary_values, position, node):
        if position == 0:
            primitiveClassification = Primitive([], "prediction", GaussianNB)
            terminalClassification = []
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)
        if position == 1:
            primitiveClassification = Primitive(['BernoulliNB.alpha', # Primitive.input
                                                 'BernoulliNB.fit_prior'],
                                                "prediction", # Primitive.output
                                                BernoulliNB # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['BernoulliNB.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['BernoulliNB.fit_prior'], 'fit_prior', 'fit_prior')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 4:
            primitiveClassification = Primitive(['MultinomialNB.alpha', # Primitive.input
                                                 'MultinomialNB.fit_prior'],
                                                "prediction", # Primitive.output
                                                MultinomialNB # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['MultinomialNB.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['MultinomialNB.fit_prior'], 'fit_prior', 'fit_prior')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 7:
            primitiveClassification = Primitive(['DecisionTreeClassifier.criterion', # Primitive.input
                                                 'DecisionTreeClassifier.max_depth',
                                                 'DecisionTreeClassifier.min_samples_split',
                                                 'DecisionTreeClassifier.min_samples_leaf'],
                                                "prediction", # Primitive.output
                                                DecisionTreeClassifier # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['DecisionTreeClassifier.criterion'], 'criterion', 'criterion')
            terminal2 = Terminal(dictionary_values['DecisionTreeClassifier.max_depth'], 'max_depth', 'max_depth')
            terminal3 = Terminal(dictionary_values['DecisionTreeClassifier.min_samples_split'], 'min_samples_split', 'min_samples_split')
            terminal4 = Terminal(dictionary_values['DecisionTreeClassifier.min_samples_leaf'], 'min_samples_leaf', 'min_samples_leaf')
            terminalClassification = [terminal1, terminal2, terminal3, terminal4]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 12:
            primitiveClassification = Primitive(['ExtraTreesClassifier.n_estimators', # Primitive.input
                                                 'ExtraTreesClassifier.criterion',
                                                 'ExtraTreesClassifier.max_features',
                                                 'ExtraTreesClassifier.min_samples_split',
                                                 'ExtraTreesClassifier.min_samples_leaf',
                                                 'ExtraTreesClassifier.bootstrap'],
                                                "prediction", # Primitive.output
                                                ExtraTreesClassifier # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['ExtraTreesClassifier.n_estimators'], 'n_estimators', 'n_estimators')
            terminal2 = Terminal(dictionary_values['ExtraTreesClassifier.criterion'], 'criterion', 'criterion')
            terminal3 = Terminal(dictionary_values['ExtraTreesClassifier.max_features'], 'max_features', 'max_features')
            terminal4 = Terminal(dictionary_values['ExtraTreesClassifier.min_samples_split'], 'min_samples_split', 'min_samples_split')
            terminal5 = Terminal(dictionary_values['ExtraTreesClassifier.min_samples_leaf'], 'min_samples_leaf', 'min_samples_leaf')
            terminal6 = Terminal(dictionary_values['ExtraTreesClassifier.bootstrap'], 'bootstrap', 'bootstrap')        
            terminalClassification = [terminal1, terminal2, terminal3, terminal4, terminal5, terminal6]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 19:
            primitiveClassification = Primitive(['RandomForestClassifier.n_estimators', # Primitive.input
                                                 'RandomForestClassifier.criterion',
                                                 'RandomForestClassifier.max_features',
                                                 'RandomForestClassifier.min_samples_split',
                                                 'RandomForestClassifier.min_samples_leaf',
                                                 'RandomForestClassifier.bootstrap'],
                                                "prediction", # Primitive.output
                                                RandomForestClassifier # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['RandomForestClassifier.n_estimators'], 'n_estimators', 'n_estimators')
            terminal2 = Terminal(dictionary_values['RandomForestClassifier.criterion'], 'criterion', 'criterion')
            terminal3 = Terminal(dictionary_values['RandomForestClassifier.max_features'], 'max_features', 'max_features')
            terminal4 = Terminal(dictionary_values['RandomForestClassifier.min_samples_split'], 'min_samples_split', 'min_samples_split')
            terminal5 = Terminal(dictionary_values['RandomForestClassifier.min_samples_leaf'], 'min_samples_leaf', 'min_samples_leaf')
            terminal6 = Terminal(dictionary_values['RandomForestClassifier.bootstrap'], 'bootstrap', 'bootstrap')        
            terminalClassification = [terminal1, terminal2, terminal3, terminal4, terminal5, terminal6]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 26:
            primitiveClassification = Primitive(['GradientBoostingClassifier.n_estimators', # Primitive.input
                                                 'GradientBoostingClassifier.learning_rate',
                                                 'GradientBoostingClassifier.max_depth',
                                                 'GradientBoostingClassifier.min_samples_split',
                                                 'GradientBoostingClassifier.min_samples_leaf',
                                                 'GradientBoostingClassifier.subsample',
                                                 'GradientBoostingClassifier.max_features'],
                                                "prediction", # Primitive.output
                                                GradientBoostingClassifier # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['GradientBoostingClassifier.n_estimators'], 'n_estimators', 'n_estimators')
            terminal2 = Terminal(dictionary_values['GradientBoostingClassifier.learning_rate'], 'learning_rate', 'learning_rate')
            terminal3 = Terminal(dictionary_values['GradientBoostingClassifier.max_depth'], 'max_depth', 'max_depth')
            terminal4 = Terminal(dictionary_values['GradientBoostingClassifier.min_samples_split'], 'min_samples_split', 'min_samples_split')
            terminal5 = Terminal(dictionary_values['GradientBoostingClassifier.min_samples_leaf'], 'min_samples_leaf', 'min_samples_leaf')
            terminal6 = Terminal(dictionary_values['GradientBoostingClassifier.subsample'], 'subsample', 'subsample')        
            terminal7 = Terminal(dictionary_values['GradientBoostingClassifier.max_features'], 'max_features', 'max_features')        
            terminalClassification = [terminal1, terminal2, terminal3, terminal4, terminal5, terminal6, terminal7]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 34:
            primitiveClassification = Primitive(['KNeighborsClassifier.n_neighbors', # Primitive.input
                                                 'KNeighborsClassifier.weights',
                                                 'KNeighborsClassifier.p'],
                                                "prediction", # Primitive.output
                                                KNeighborsClassifier # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['KNeighborsClassifier.n_neighbors'], 'n_neighbors', 'n_neighbors')
            terminal2 = Terminal(dictionary_values['KNeighborsClassifier.weights'], 'weights', 'weights')
            terminal3 = Terminal(dictionary_values['KNeighborsClassifier.p'], 'p', 'p')
            terminalClassification = [terminal1, terminal2, terminal3]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 38:
            primitiveClassification = Primitive(['LinearSVC.penalty', # Primitive.input
                                                 'LinearSVC.loss',
                                                 'LinearSVC.dual',
                                                 'LinearSVC.tol',
                                                 'LinearSVC.C'],
                                                "prediction", # Primitive.output
                                                LinearSVC # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['LinearSVC.penalty'], 'penalty', 'penalty')
            terminal2 = Terminal(dictionary_values['LinearSVC.loss'], 'loss', 'loss')
            terminal3 = Terminal(dictionary_values['LinearSVC.dual'], 'dual', 'dual')
            terminal4 = Terminal(dictionary_values['LinearSVC.tol'], 'tol', 'tol')
            terminal5 = Terminal(dictionary_values['LinearSVC.C'], 'C', 'C')
            terminalClassification = [terminal1, terminal2, terminal3, terminal4, terminal5]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 44:
            primitiveClassification = Primitive(['LogisticRegression.penalty', # Primitive.input
                                                 'LogisticRegression.C',
                                                 'LogisticRegression.dual',
                                                 'LogisticRegression.solver'],
                                                "prediction", # Primitive.output
                                                LogisticRegression # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['LogisticRegression.penalty'], 'penalty', 'penalty')
            terminal2 = Terminal(dictionary_values['LogisticRegression.C'], 'C', 'C')
            terminal3 = Terminal(dictionary_values['LogisticRegression.dual'], 'dual', 'dual')
            terminal4 = Terminal(dictionary_values['LogisticRegression.solver'], 'solver', 'solver')
            terminalClassification = [terminal1, terminal2, terminal3, terminal4]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 49:
            primitiveClassification = Primitive(['Binarizer.threshold'],
                                                "data", # Primitive.output
                                                Binarizer # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Binarizer.threshold'], 'threshold', 'threshold')
            terminalClassification = [terminal1]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 51:
            primitiveClassification = Primitive(['FastICA.tol'],
                                                "data", # Primitive.output
                                                FastICA # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['FastICA.tol'], 'tol', 'tol')
            terminalClassification = [terminal1]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 53:
            primitiveClassification = Primitive(['FeatureAgglomeration.linkage',
                                                'FeatureAgglomeration.affinity'],
                                                "data", # Primitive.output
                                                FeatureAgglomeration # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['FeatureAgglomeration.linkage'], 'linkage', 'linkage')
            terminal2 = Terminal(dictionary_values['FeatureAgglomeration.affinity'], 'affinity', 'affinity')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 56:
            primitiveClassification = Primitive([],"data", MaxAbsScaler)
            terminalClassification = []
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 57:
            primitiveClassification = Primitive([],"data", MinMaxScaler)
            terminalClassification = []
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 58:
            primitiveClassification = Primitive(['Normalizer.norm'],
                                                "data", # Primitive.output
                                                Normalizer # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Normalizer.norm'], 'norm', 'norm')
            terminalClassification = [terminal1]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 60:
            primitiveClassification = Primitive(['Nystroem.kernel',
                                                'Nystroem.gamma',
                                                'Nystroem.n_components'],
                                                "data", # Primitive.output
                                                Nystroem # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Nystroem.kernel'], 'kernel', 'kernel')
            terminal2 = Terminal(dictionary_values['Nystroem.gamma'], 'gamma', 'gamma')
            terminal3 = Terminal(dictionary_values['Nystroem.n_components'], 'n_components', 'n_components')
            terminalClassification = [terminal1, terminal2, terminal3]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 64:
            primitiveClassification = Primitive(['PCA.svd_solver',
                                                'PCA.iterated_power'],
                                                "data", # Primitive.output
                                                PCA # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['PCA.svd_solver'], 'svd_solver', 'svd_solver')
            terminal2 = Terminal(dictionary_values['PCA.iterated_power'], 'iterated_power', 'iterated_power')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 67:
            primitiveClassification = Primitive(['PolynomialFeatures.degree',
                                                'PolynomialFeatures.include_bias',
                                                'PolynomialFeatures.interaction_only'],
                                                "data", # Primitive.output
                                                PolynomialFeatures # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['PolynomialFeatures.degree'], 'degree', 'degree')
            terminal2 = Terminal(dictionary_values['PolynomialFeatures.include_bias'], 'include_bias', 'include_bias')
            terminal3 = Terminal(dictionary_values['PolynomialFeatures.interaction_only'], 'interaction_only', 'interaction_only')
            terminalClassification = [terminal1, terminal2, terminal3]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 71:
            primitiveClassification = Primitive(['RBFSampler.gamma'],
                                                "data", # Primitive.output
                                                RBFSampler # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['RBFSampler.gamma'], 'gamma', 'gamma')
            terminalClassification = [terminal1]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 73:
            primitiveClassification = Primitive([],"data", RobustScaler)
            terminalClassification = []
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 74:
            primitiveClassification = Primitive([],"data", StandardScaler)
            terminalClassification = []
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 75:
            primitiveClassification = Primitive(['SelectFwe.alpha',
                                                'SelectFwe.score_func'],
                                                "data", # Primitive.output
                                                SelectFwe # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['SelectFwe.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['SelectFwe.score_func'], 'score_func', 'score_func')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 77:
            primitiveClassification = Primitive(['SelectPercentile.percentile',
                                                'SelectPercentile.score_func'],
                                                "data", # Primitive.output
                                                SelectPercentile # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['SelectPercentile.percentile'], 'percentile', 'percentile')
            terminal2 = Terminal(dictionary_values['SelectPercentile.score_func'], 'score_func', 'score_func')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)            
        if position == 79:
            primitiveClassification = Primitive(['VarianceThreshold.threshold'],
                                                "data", # Primitive.output
                                                VarianceThreshold # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['VarianceThreshold.threshold'], 'threshold', 'threshold')
            terminalClassification = [terminal1]
            primitiveNodeClassification= PrimitiveNode(primitiveClassification, node, terminalClassification)              
        return primitiveNodeClassification
    
    
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

class IndividuoVector(object):
    def __init__(self, upperBound=upperBound, lowerBound=lowerBound, vector_support=vector_support):
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.vector_support = vector_support
        vector_to_return = []
        for i in range(len(self.upperBound)):
            if self.vector_support[i] == 1:
                vector_to_return.append(np.random.uniform(30,50))
            else:
                vector_to_return.append(np.random.uniform(self.lowerBound[i],self.upperBound[i]))
        self.vector_to_return = vector_to_return
        self.contador_classification = 0
        
    def __call__(self, individuo_main):
        return self.convert(individuo_main.main_node)
    
    
    def convert(self, primitive_node_part):
        self.contador_classification += 1
        #print("self.contador_classification", self.contador_classification)
        _primitive = primitive_node_part._primitive 
        _terminals = primitive_node_part._terminals
        self.vector_to_return = self.choose_values(_primitive, _terminals)
        #print(self.vector_to_return )
        if isinstance(primitive_node_part._data_node, PrimitiveNode):
            #print("primitive_node_part._data_node", primitive_node_part._data_node)
            self.convert(primitive_node_part._data_node)
            self.contador_classification += 1 
        return self.vector_to_return
                
    # def convert(self, individuo_main):
    #     print("type", type(individuo_main))
    #     contador_classification = 1
    #     print("contador_classification", contador_classification)
    #     primitive_node = individuo_main.main_node
    #     _primitive = primitive_node._primitive 
    #     _terminals = primitive_node._terminals
    #     vector = self.choose_values(_primitive, _terminals, contador_classification)
    #     while isinstance(primitive_node._data_node, PrimitiveNode):
    #         self.convert(primitive_node._data_node)
    #         contador_classification += 1
    #     return vector
    
    # def convert(self, primitive_part):
    #     print("self.contador_classification", self.contador_classification)
    #     _primitive = primitive_part._primitive 
    #     _terminals = primitive_part._terminals
    #     vector = self.choose_values(_primitive, _terminals)
    #     while isinstance(primitive_part._data_node, PrimitiveNode):
    #         self.convert(primitive_part)
    #         self.contador_classification += 1
    #     return vector
           
    def choose_values(self, primitive_object, terminal_object):
        #contador = self.contador_classification
        contador = self.contador_classification
        #if KNeighborsClassifier == ejem1.main_node._primitive.identifier:
        if isinstance(primitive_object.identifier(), GaussianNB):
            self.vector_to_return[0] = 100 - contador
            
        if isinstance(primitive_object.identifier(), BernoulliNB):
            self.vector_to_return[1] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'alpha':
                    self.vector_to_return[2] = terminal_object[i].value
                if terminal_object[i].output == 'fit_prior':
                    self.vector_to_return[3] = _boo_to_int(terminal_object[i].value)
                    
        if isinstance(primitive_object.identifier(), MultinomialNB):
            self.vector_to_return[4] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'alpha':
                    self.vector_to_return[5] = terminal_object[i].value
                if terminal_object[i].output == 'fit_prior':
                    self.vector_to_return[6] = _boo_to_int(terminal_object[i].value)
                    
        if isinstance(primitive_object.identifier(), DecisionTreeClassifier):
            self.vector_to_return[7] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'criterion':
                    self.vector_to_return[8] = _string_to_int(terminal_object[i].value, gini=0, entropy=1)
                if terminal_object[i].output == 'max_depth':
                    self.vector_to_return[9] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_split':
                    self.vector_to_return[10] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_leaf':
                    self.vector_to_return[11] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), ExtraTreesClassifier):
            self.vector_to_return[12] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'n_estimators':
                    self.vector_to_return[13] = terminal_object[i].value
                if terminal_object[i].output == 'criterion':
                    self.vector_to_return[14] = _string_to_int(terminal_object[i].value, gini=0, entropy=1)
                if terminal_object[i].output == 'max_features':
                    self.vector_to_return[15] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_split':
                    self.vector_to_return[16] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_leaf':
                    self.vector_to_return[17] = terminal_object[i].value
                if terminal_object[i].output == 'bootstrap':
                    self.vector_to_return[18] = _boo_to_int(terminal_object[i].value)
                    
        if isinstance(primitive_object.identifier(), RandomForestClassifier):
            self.vector_to_return[19] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'n_estimators':
                    self.vector_to_return[20] = terminal_object[i].value
                if terminal_object[i].output == 'criterion':
                    self.vector_to_return[21] = _string_to_int(terminal_object[i].value, gini=0, entropy=1)
                if terminal_object[i].output == 'max_features':
                    self.vector_to_return[22] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_split':
                    self.vector_to_return[23] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_leaf':
                    self.vector_to_return[24] = terminal_object[i].value
                if terminal_object[i].output == 'bootstrap':
                    self.vector_to_return[25] = _boo_to_int(terminal_object[i].value)
                    
        if isinstance(primitive_object.identifier(), GradientBoostingClassifier):
            self.vector_to_return[26] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'n_estimators':
                    self.vector_to_return[27] = terminal_object[i].value
                if terminal_object[i].output == 'learning_rate':
                    self.vector_to_return[28] = terminal_object[i].value
                if terminal_object[i].output == 'max_depth':
                    self.vector_to_return[29] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_split':
                    self.vector_to_return[30] = terminal_object[i].value
                if terminal_object[i].output == 'min_samples_leaf':
                    self.vector_to_return[31] = terminal_object[i].value
                if terminal_object[i].output == 'subsample':
                    self.vector_to_return[32] = terminal_object[i].value
                if terminal_object[i].output == 'max_features':
                    self.vector_to_return[33] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), KNeighborsClassifier):
            self.vector_to_return[34] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'n_neighbors':
                    self.vector_to_return[35] = terminal_object[i].value
                if terminal_object[i].output == 'weights':
                    self.vector_to_return[36] = _string_to_int(terminal_object[i].value, uniform=0, distance=1)
                if terminal_object[i].output == 'p':
                    self.vector_to_return[37] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), LinearSVC):
            self.vector_to_return[38] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'penalty':
                    self.vector_to_return[39] = _string_to_int(terminal_object[i].value, l1=0, l2=1)
                if terminal_object[i].output == 'loss':
                    self.vector_to_return[40] = _string_to_int(terminal_object[i].value, hinge=0, squared_hinge=1)
                if terminal_object[i].output == 'dual':
                    self.vector_to_return[41] = _boo_to_int(terminal_object[i].value)
                if terminal_object[i].output == 'tol':
                    self.vector_to_return[42] = terminal_object[i].value
                if terminal_object[i].output == 'C':
                    self.vector_to_return[43] = terminal_object[i].value
            
        if isinstance(primitive_object.identifier(), LogisticRegression):
            self.vector_to_return[44] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'penalty':
                    self.vector_to_return[45] = _string_to_int(terminal_object[i].value, l2=0)
                if terminal_object[i].output == 'C':
                    self.vector_to_return[46] = terminal_object[i].value
                if terminal_object[i].output == 'dual':
                    self.vector_to_return[47] = _boo_to_int(terminal_object[i].value)
                if terminal_object[i].output == 'solver':
                    self.vector_to_return[48] = _string_to_int(terminal_object[i].value, lbfgs=0)
                    
        if isinstance(primitive_object.identifier(), Binarizer):
            self.vector_to_return[49] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'threshold':
                    self.vector_to_return[50] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), FastICA):
            self.vector_to_return[51] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'tol':
                    self.vector_to_return[52] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), FeatureAgglomeration):
           self.vector_to_return[53] = 100 - contador
           for i in range(len(terminal_object)):
               if terminal_object[i].output == 'linkage':
                   self.vector_to_return[54] = _string_to_int(terminal_object[i].value, ward=0, complete=1, average=2)
               if terminal_object[i].output == 'affinity':
                   self.vector_to_return[55] = _string_to_int(terminal_object[i].value, 
                                                         euclidean=0, 
                                                         l1=1, 
                                                         l2=2,
                                                         manhattan=3,
                                                         cosine=4,
                                                         precomputed=5)
       
        if isinstance(primitive_object.identifier(), MaxAbsScaler):
            self.vector_to_return[56] = 100 - contador
            
        if isinstance(primitive_object.identifier(), MinMaxScaler):
            self.vector_to_return[57] = 100 - contador
            
        if isinstance(primitive_object.identifier(), Normalizer):
            self.vector_to_return[58] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'norm':
                    self.vector_to_return[59] = _string_to_int(terminal_object[i].value, l1=0, l2=1, max=2)
            
        if isinstance(primitive_object.identifier(), Nystroem):
            self.vector_to_return[60] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'kernel':
                    self.vector_to_return[61] = _string_to_int(terminal_object[i].value, 
                                                          rbf=0, 
                                                          cosine=1, 
                                                          chi2=2, 
                                                          laplacian=3, 
                                                          polynomial=4, 
                                                          poly=5, 
                                                          linear=6, 
                                                          additive_chi2=7, 
                                                          sigmoid=8)
                if terminal_object[i].output == 'gamma':
                    self.vector_to_return[62] = terminal_object[i].value
                if terminal_object[i].output == 'n_components':
                    self.vector_to_return[63] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), PCA):
            self.vector_to_return[64] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'svd_solver':
                    self.vector_to_return[65] = _string_to_int(terminal_object[i].value, randomized=0)
                if terminal_object[i].output == 'iterated_power':
                    self.vector_to_return[66] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), PolynomialFeatures):
            self.vector_to_return[67] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'degree':
                    self.vector_to_return[68] = terminal_object[i].value
                if terminal_object[i].output == 'include_bias':
                    self.vector_to_return[69] = _boo_to_int(terminal_object[i].value)
                if terminal_object[i].output == 'interaction_only':
                    self.vector_to_return[70] = _boo_to_int(terminal_object[i].value)
                    
        if isinstance(primitive_object.identifier(), RBFSampler):
            self.vector_to_return[71] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'gamma':
                    self.vector_to_return[72] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), RobustScaler):
            self.vector_to_return[73] = 100 - contador
            
        if isinstance(primitive_object.identifier(), StandardScaler):
            self.vector_to_return[74] = 100 - contador
            
        if isinstance(primitive_object.identifier(), SelectFwe):
            self.vector_to_return[75] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'alpha':
                    self.vector_to_return[76] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), SelectPercentile):
            self.vector_to_return[77] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'percentile':
                    self.vector_to_return[78] = terminal_object[i].value
                    
        if isinstance(primitive_object.identifier(), VarianceThreshold):
            self.vector_to_return[79] = 100 - contador
            for i in range(len(terminal_object)):
                if terminal_object[i].output == 'threshold':
                    self.vector_to_return[80] = terminal_object[i].value
                    
        return self.vector_to_return
      
lista_ind = []             
for i in export:
    newInstance = ValuesSearchSpace(i)
    result1 = newInstance.get_individuals()
    lista_ind.append(result1)
    

nueva_lista_v = []
for i in lista_ind:
    instance_try = IndividuoVector()
    new_v1 = instance_try(i)
    nueva_lista_v.append(new_v1)
 
lista_to_compare = []
for h in nueva_lista_v:
    newInstance = ValuesSearchSpace(h)
    result1 = newInstance.get_individuals()
    lista_to_compare.append(result1)
    
for i in range(len(export)):
    print("REAL")
    print(lista_ind[i])
    print("FICTICIO")
    print(lista_to_compare[i])
    
# ejem1 = lista_ind[0]
# instance_try = IndividuoVector()
# ay = instance_try(ejem1)

# newInstance2 = ValuesSearchSpace(ay)
# result2 = newInstance2.get_individuals()
# print(result2)


# ejem2 = lista_ind[1]
# instance_try = IndividuoVector()
# ay = instance_try(ejem2)

# newInstance3 = ValuesSearchSpace(ay)
# result3 = newInstance3.get_individuals()
# print(result3)


# lista_con_new_vectors = []
# for i in lista_ind:
#     instance_try = IndividuoVector()
#     new_v = instance_try(i)
#     lista_con_new_vectors.append(new_v)
# vector_to_return = [0] * len(export[0])
        # ejem1.main_node._terminals
        # [n_neighbors=49, p=1, weights='uniform']
        
        # ejem1.main_node._terminals[0]
        # n_neighbors=49
        
        
        # ejem1.main_node._terminals[0].value
        # 49
        
        # ejem1.main_node._terminals[0].identifier # .output si no funciona
        # 'n_neighbors'
        
        # ejem1.main_node._terminals[1].identifier
        # 'p'
        
        # ejem1.main_node._terminals[1].value
        # 1
    