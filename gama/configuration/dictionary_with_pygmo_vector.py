# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:06:51 2021

@author: 20210595
"""
from space_to_vector2 import export, positions, positionClassifier, positionPreprocess
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
        dictionary_pygmo.update({'SelectFwe.score_func': {f_classif: None}})
        dictionary_pygmo.update({'SelectPercentile': self.x[77]})
        dictionary_pygmo.update({'SelectPercentile.percentile': round(self.x[78])})
        dictionary_pygmo.update({'SelectPercentile.score_func': {f_classif: None}})
        dictionary_pygmo.update({'VarianceThreshold': self.x[79]})
        dictionary_pygmo.update({'VarianceThreshold.threshold': self.x[80]})
        
        newpositions = self._index_function(x=self.x)
        #individual_p = self._create_individual(dictionary_values=dictionary_pygmo, position=newpositions[0])
        return newpositions
    
    def _int_to_string(self, value, **kwargs):
        for element in kwargs:
            if kwargs[element] == value:
                return element 
            
    def _int_to_bool(self, value):
        return True if value == 1 else False
    
    def _index_function(self, x):            
        list_index_techniques_to_use_before = [i for i in positionPreprocess if x[i] > 80]
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
        #The first index is the classifier, position 0
        list_index_techniques_to_use.insert(0, indexClassifier)
        #list_index_techniques_to_use.append(indexClassifier)
        return list_index_techniques_to_use
 
    def _create_individual(self, dictionary_values, position):
        if position == 0:
            primitiveClassification = Primitive([], "prediction", GaussianNB)
            terminalClassification = []
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)
            ind = Individual(primitiveNodeClassification, compile_individual)
        if position == 1:
            primitiveClassification = Primitive(['BernoulliNB.alpha', # Primitive.input
                                                 'BernoulliNB.fit_prior'],
                                                "prediction", # Primitive.output
                                                BernoulliNB # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['BernoulliNB.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['BernoulliNB.fit_prior'], 'fit_prior', 'fit_prior')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
        if position == 4:
            primitiveClassification = Primitive(['MultinomialNB.alpha', # Primitive.input
                                                 'MultinomialNB.fit_prior'],
                                                "prediction", # Primitive.output
                                                MultinomialNB # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['MultinomialNB.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['MultinomialNB.fit_prior'], 'fit_prior', 'fit_prior')
            terminalClassification = [terminal1, terminal2]
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
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
            primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)            
            ind = Individual(primitiveNodeClassification, compile_individual)
        if position == 49:
            primitivePreprocessing = Primitive(['Binarizer.threshold'],
                                                "data", # Primitive.output
                                                Binarizer # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Binarizer.threshold'], 'threshold', 'threshold')
            terminalPreprocessing = [terminal1]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 51:
            primitivePreprocessing = Primitive(['FastICA.tol'],
                                                "data", # Primitive.output
                                                FastICA # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['FastICA.tol'], 'tol', 'tol')
            terminalPreprocessing = [terminal1]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 53:
            primitivePreprocessing = Primitive(['FeatureAgglomeration.linkage',
                                                'FeatureAgglomeration.affinity'],
                                                "data", # Primitive.output
                                                FeatureAgglomeration # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['FeatureAgglomeration.linkage'], 'linkage', 'linkage')
            terminal2 = Terminal(dictionary_values['FeatureAgglomeration.affinity'], 'affinity', 'affinity')
            terminalPreprocessing = [terminal1, terminal2]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 56:
            primitivePreprocessing = Primitive([],"data", MaxAbsScaler)
            terminalPreprocessing = []
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 57:
            primitivePreprocessing = Primitive([],"data", MinMaxScaler)
            terminalPreprocessing = []
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 58:
            primitivePreprocessing = Primitive(['Normalizer.norm'],
                                                "data", # Primitive.output
                                                Normalizer # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Normalizer.norm'], 'norm', 'norm')
            terminalPreprocessing = [terminal1]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 60:
            primitivePreprocessing = Primitive(['Nystroem.kernel',
                                                'Nystroem.gamma',
                                                'Nystroem.n_components'],
                                                "data", # Primitive.output
                                                Nystroem # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['Nystroem.kernel'], 'kernel', 'kernel')
            terminal2 = Terminal(dictionary_values['Nystroem.gamma'], 'gamma', 'gamma')
            terminal3 = Terminal(dictionary_values['Nystroem.n_components'], 'n_components', 'n_components')
            terminalPreprocessing = [terminal1, terminal2, terminal3]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 64:
            primitivePreprocessing = Primitive(['PCA.svd_solver',
                                                'PCA.iterated_power'],
                                                "data", # Primitive.output
                                                PCA # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['PCA.svd_solver'], 'svd_solver', 'svd_solver')
            terminal2 = Terminal(dictionary_values['PCA.iterated_power'], 'iterated_power', 'iterated_power')
            terminalPreprocessing = [terminal1, terminal2]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 67:
            primitivePreprocessing = Primitive(['PolynomialFeatures.degree',
                                                'PolynomialFeatures.include_bias',
                                                'PolynomialFeatures.interaction_only'],
                                                "data", # Primitive.output
                                                PolynomialFeatures # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['PolynomialFeatures.degree'], 'degree', 'degree')
            terminal2 = Terminal(dictionary_values['PolynomialFeatures.include_bias'], 'include_bias', 'include_bias')
            terminal3 = Terminal(dictionary_values['PolynomialFeatures.interaction_only'], 'interaction_only', 'interaction_only')
            terminalPreprocessing = [terminal1, terminal2, terminal3]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 71:
            primitivePreprocessing = Primitive(['RBFSampler.gamma'],
                                                "data", # Primitive.output
                                                RBFSampler # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['RBFSampler.gamma'], 'gamma', 'gamma')
            terminalPreprocessing = [terminal1]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 73:
            primitivePreprocessing = Primitive([],"data", RobustScaler)
            terminalPreprocessing = []
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 74:
            primitivePreprocessing = Primitive([],"data", StandardScaler)
            terminalPreprocessing = []
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 75:
            primitivePreprocessing = Primitive(['SelectFwe.alpha',
                                                'SelectFwe.score_func'],
                                                "data", # Primitive.output
                                                SelectFwe # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['SelectFwe.alpha'], 'alpha', 'alpha')
            terminal2 = Terminal(dictionary_values['SelectFwe.score_func'], 'score_func', 'score_func')
            terminalPreprocessing = [terminal1, terminal2]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 77:
            primitivePreprocessing = Primitive(['SelectPercentile.percentile',
                                                'SelectPercentile.score_func'],
                                                "data", # Primitive.output
                                                SelectPercentile # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['SelectPercentile.percentile'], 'percentile', 'percentile')
            terminal2 = Terminal(dictionary_values['SelectPercentile.score_func'], 'score_func', 'score_func')
            terminalPreprocessing = [terminal1, terminal2]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
        if position == 79:
            primitivePreprocessing = Primitive(['VarianceThreshold.threshold'],
                                                "data", # Primitive.output
                                                VarianceThreshold # Primitive.identifier
                                                )
            terminal1 = Terminal(dictionary_values['VarianceThreshold.threshold'], 'threshold', 'threshold')
            terminalPreprocessing = [terminal1]
            primitiveNodePreprocessing= PrimitiveNode(primitivePreprocessing, "data", terminalPreprocessing)            
            ind = Individual(primitiveNodePreprocessing, compile_individual)
            
for i in export:
    newInstance = ValuesSearchSpace(i)
    result1 = newInstance.get_individuals()
    print(result1)
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# X, y = make_classification(n_features=4, random_state=0)
# clf = make_pipeline(StandardScaler(),
#                     LinearSVC(penalty='l2', loss='hinge', dual=True))
# clf.fit(X, y)
