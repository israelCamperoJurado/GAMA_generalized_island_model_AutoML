# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 00:58:52 2021

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

from sklearn.pipeline import make_pipeline, Pipeline
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
    def __init__(self):
        self.counter = 0
        
    def get_individuals(self, x):
        dictionary_pygmo = {}
        dictionary_pygmo.update({'GaussianNB': x[0]})
        dictionary_pygmo.update({'BernoulliNB': x[1]})
        dictionary_pygmo.update({'BernoulliNB.alpha': x[2]})
        dictionary_pygmo.update({'BernoulliNB.fit_prior': self._int_to_bool(round(x[3]))})
        dictionary_pygmo.update({'MultinomialNB': x[4]})
        dictionary_pygmo.update({'MultinomialNB.alpha': x[5]})
        dictionary_pygmo.update({'MultinomialNB.fit_prior': self._int_to_bool(round(x[6]))})
        dictionary_pygmo.update({'DecisionTreeClassifier': x[7]})
        dictionary_pygmo.update({'DecisionTreeClassifier.criterion': self._int_to_string(round(x[8]), gini=0, entropy=1)})
        dictionary_pygmo.update({'DecisionTreeClassifier.max_depth': round(x[9])})
        dictionary_pygmo.update({'DecisionTreeClassifier.min_samples_split': round(x[10])})
        dictionary_pygmo.update({'DecisionTreeClassifier.min_samples_leaf': round(x[11])})
        dictionary_pygmo.update({'ExtraTreesClassifier': x[12]})
        dictionary_pygmo.update({'ExtraTreesClassifier.n_estimators': round(x[13])})
        dictionary_pygmo.update({'ExtraTreesClassifier.criterion': self._int_to_string(round(x[14]), gini=0, entropy=1)})
        dictionary_pygmo.update({'ExtraTreesClassifier.max_features': x[15]})
        dictionary_pygmo.update({'ExtraTreesClassifier.min_samples_split': round(x[16])})
        dictionary_pygmo.update({'ExtraTreesClassifier.min_samples_leaf': round(x[17])})
        dictionary_pygmo.update({'ExtraTreesClassifier.bootstrap': self._int_to_bool(round(x[18]))})
        dictionary_pygmo.update({'RandomForestClassifier': x[19]})
        dictionary_pygmo.update({'RandomForestClassifier.n_estimators': round(x[20])})
        dictionary_pygmo.update({'RandomForestClassifier.criterion': self._int_to_string(round(x[21]), gini=0, entropy=1)})
        dictionary_pygmo.update({'RandomForestClassifier.max_features': x[22]})
        dictionary_pygmo.update({'RandomForestClassifier.min_samples_split': round(x[23])})
        dictionary_pygmo.update({'RandomForestClassifier.min_samples_leaf': round(x[24])})
        dictionary_pygmo.update({'RandomForestClassifier.bootstrap': self._int_to_bool(round(x[25]))})
        dictionary_pygmo.update({'GradientBoostingClassifier': x[26]})
        dictionary_pygmo.update({'GradientBoostingClassifier.n_estimators': round(x[27])})
        dictionary_pygmo.update({'GradientBoostingClassifier.learning_rate': x[28]})
        dictionary_pygmo.update({'GradientBoostingClassifier.max_depth': round(x[29])})
        dictionary_pygmo.update({'GradientBoostingClassifier.min_samples_split': round(x[30])})
        dictionary_pygmo.update({'GradientBoostingClassifier.min_samples_leaf': round(x[31])})
        dictionary_pygmo.update({'GradientBoostingClassifier.subsample': x[32]})
        dictionary_pygmo.update({'GradientBoostingClassifier.max_features': x[33]})
        dictionary_pygmo.update({'KNeighborsClassifier': x[34]})
        dictionary_pygmo.update({'KNeighborsClassifier.n_neighbors': round(x[35])})
        dictionary_pygmo.update({'KNeighborsClassifier.weights': self._int_to_string(round(x[36]), uniform=0, distance=1)})
        dictionary_pygmo.update({'KNeighborsClassifier.p': round(x[37])})
        dictionary_pygmo.update({'LinearSVC': x[38]})
        dictionary_pygmo.update({'LinearSVC.penalty': self._int_to_string(round(x[39]), l1=0, l2=1)})
        dictionary_pygmo.update({'LinearSVC.loss': self._int_to_string(round(x[40]), hinge=0, squared_hinge=1)})
        dictionary_pygmo.update({'LinearSVC.dual': self._int_to_bool(round(x[41]))})
        dictionary_pygmo.update({'LinearSVC.tol': x[42]})
        dictionary_pygmo.update({'LinearSVC.C': x[43]})
        if dictionary_pygmo['LinearSVC.penalty'] == 'l1':
            dictionary_pygmo['LinearSVC.loss'] = 'squared_hinge'
        if (dictionary_pygmo['LinearSVC.penalty'] == 'l2') and (dictionary_pygmo['LinearSVC.loss'] == 'hinge') and (dictionary_pygmo['LinearSVC.dual'] == False):
            dictionary_pygmo['LinearSVC.dual'] == True
        if (dictionary_pygmo['LinearSVC.penalty'] == 'l1') and (dictionary_pygmo['LinearSVC.loss'] == 'squared_hinge') and (dictionary_pygmo['LinearSVC.dual'] == True):
            dictionary_pygmo['LinearSVC.dual'] == False
        dictionary_pygmo.update({'LogisticRegression': x[44]})
        dictionary_pygmo.update({'LogisticRegression.penalty': self._int_to_string(round(x[45]), l2=0)})
        dictionary_pygmo.update({'LogisticRegression.C': x[46]})
        dictionary_pygmo.update({'LogisticRegression.dual': self._int_to_bool(round(x[47]))})
        dictionary_pygmo.update({'LogisticRegression.solver': self._int_to_string(round(x[48]), lbfgs=0)})
        dictionary_pygmo.update({'Binarizer': x[49]})
        dictionary_pygmo.update({'Binarizer.threshold': x[50]})
        dictionary_pygmo.update({'FastICA': x[51]})
        dictionary_pygmo.update({'FastICA.tol': x[52]})
        dictionary_pygmo.update({'FeatureAgglomeration': x[53]})
        dictionary_pygmo.update({'FeatureAgglomeration.linkage': self._int_to_string(round(x[54]), ward=0, complete=1, average=2)})
        dictionary_pygmo.update({'FeatureAgglomeration.affinity': self._int_to_string(round(x[55]), euclidean=0, l1=1, l2=2, manhattan=3, cosine=4, precomputed=5)})
        if dictionary_pygmo['FeatureAgglomeration.linkage'] == 'ward':
            dictionary_pygmo['FeatureAgglomeration.affinity'] = 'euclidean'
        dictionary_pygmo.update({'MaxAbsScaler': x[56]})
        dictionary_pygmo.update({'MinMaxScaler': x[57]})
        dictionary_pygmo.update({'Normalizer': x[58]})
        dictionary_pygmo.update({'Normalizer.norm': self._int_to_string(round(x[59]), l1=0, l2=1, max=2)})
        dictionary_pygmo.update({'Nystroem': x[60]})
        dictionary_pygmo.update({'Nystroem.kernel': self._int_to_string(round(x[61]), rbf=0, cosine=1, chi2=2, laplacian=3, polynomial=4, poly=5, linear=6, additive_chi2=7, sigmoid=8)})
        dictionary_pygmo.update({'Nystroem.gamma': x[62]})
        dictionary_pygmo.update({'Nystroem.n_components': round(x[63])})
        dictionary_pygmo.update({'PCA': x[64]})
        dictionary_pygmo.update({'PCA.svd_solver': self._int_to_string(round(x[65]), randomized=0)})
        dictionary_pygmo.update({'PCA.iterated_power': round(x[66])})
        dictionary_pygmo.update({'PolynomialFeatures': x[67]})
        dictionary_pygmo.update({'PolynomialFeatures.degree': round(x[68])})
        dictionary_pygmo.update({'PolynomialFeatures.include_bias': self._int_to_bool(round(x[69]))})
        dictionary_pygmo.update({'PolynomialFeatures.interaction_only': self._int_to_bool(round(x[70]))})
        dictionary_pygmo.update({'RBFSampler': x[71]})
        dictionary_pygmo.update({'RBFSampler.gamma': x[72]})
        dictionary_pygmo.update({'RobustScaler': x[73]})
        dictionary_pygmo.update({'StandardScaler': x[74]})
        dictionary_pygmo.update({'SelectFwe': x[75]})
        dictionary_pygmo.update({'SelectFwe.alpha': x[76]})
        #dictionary_pygmo.update({'SelectFwe.score_func': {f_classif: None}})
        dictionary_pygmo.update({'SelectFwe.score_func': f_classif})
        dictionary_pygmo.update({'SelectPercentile': x[77]})
        dictionary_pygmo.update({'SelectPercentile.percentile': round(x[78])})
        #dictionary_pygmo.update({'SelectPercentile.score_func': {f_classif: None}})
        dictionary_pygmo.update({'SelectPercentile.score_func': f_classif})
        dictionary_pygmo.update({'VarianceThreshold': x[79]})
        dictionary_pygmo.update({'VarianceThreshold.threshold': x[80]})
        
        newpositions = self._index_function(x, dictionary_pygmo)
        #pipeline = self._create_pipeline(dictionary_values=dictionary_pygmo, position=newpositions[0])
        return newpositions
    
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
        lista_of_estimators = [self._create_individual(dictionary_pos, i) for i in list_index_techniques_to_use]
        #return ind, list_index_techniques_to_use
        clf = Pipeline(lista_of_estimators)
        return clf
 
    def _create_individual(self, dictionary_values, position):
        if position == 0:
            estimator = ('GaussianNB', GaussianNB())
        if position == 1:
            estimator = ('BernoulliNB', BernoulliNB(alpha=dictionary_values['BernoulliNB.alpha'],
                                                    fit_prior = dictionary_values['BernoulliNB.fit_prior']))        
        if position == 4:
            estimator = ('MultinomialNB', MultinomialNB(alpha=dictionary_values['MultinomialNB.alpha'],
                                                        fit_prior = dictionary_values['MultinomialNB.fit_prior']))             
        if position == 7:
            estimator = ('DecisionTreeClassifier', DecisionTreeClassifier(criterion=dictionary_values['DecisionTreeClassifier.criterion'],
                                                                          max_depth=dictionary_values['DecisionTreeClassifier.max_depth'],
                                                                          min_samples_split=dictionary_values['DecisionTreeClassifier.min_samples_split'],
                                                                          min_samples_leaf=dictionary_values['DecisionTreeClassifier.min_samples_leaf']))     
        if position == 12:
            estimator = ('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=dictionary_values['ExtraTreesClassifier.n_estimators'],
                                                                      criterion=dictionary_values['ExtraTreesClassifier.criterion'],
                                                                      max_features=dictionary_values['ExtraTreesClassifier.max_features'],
                                                                      min_samples_split=dictionary_values['ExtraTreesClassifier.min_samples_split'],
                                                                      min_samples_leaf=dictionary_values['ExtraTreesClassifier.min_samples_leaf'],
                                                                      bootstrap=dictionary_values['ExtraTreesClassifier.bootstrap'])) 
           
        if position == 19:
            estimator = ('RandomForestClassifier', RandomForestClassifier(n_estimators=dictionary_values['RandomForestClassifier.n_estimators'],
                                                                          criterion=dictionary_values['RandomForestClassifier.criterion'],
                                                                          max_features=dictionary_values['RandomForestClassifier.max_features'],
                                                                          min_samples_split=dictionary_values['RandomForestClassifier.min_samples_split'],
                                                                          min_samples_leaf=dictionary_values['RandomForestClassifier.min_samples_leaf'],
                                                                          bootstrap=dictionary_values['RandomForestClassifier.bootstrap'])) 
          
        if position == 26:
            estimator = ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=dictionary_values['GradientBoostingClassifier.n_estimators'],
                                                                                  learning_rate=dictionary_values['GradientBoostingClassifier.learning_rate'],
                                                                                  max_depth=dictionary_values['GradientBoostingClassifier.max_depth'],
                                                                                  min_samples_split=dictionary_values['GradientBoostingClassifier.min_samples_split'],
                                                                                  min_samples_leaf=dictionary_values['GradientBoostingClassifier.min_samples_leaf'],
                                                                                  subsample=dictionary_values['GradientBoostingClassifier.subsample'],
                                                                                  max_features=dictionary_values['GradientBoostingClassifier.max_features']))
        
        if position == 34:
            estimator = ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=dictionary_values['KNeighborsClassifier.n_neighbors'],
                                                                      weights=dictionary_values['KNeighborsClassifier.weights'],
                                                                      p=dictionary_values['KNeighborsClassifier.p']))
          
        if position == 38:
            estimator = ('LinearSVC', LinearSVC(penalty=dictionary_values['LinearSVC.penalty'],
                                                loss=dictionary_values['LinearSVC.loss'],
                                                dual=dictionary_values['LinearSVC.dual'],
                                                tol=dictionary_values['LinearSVC.tol'],
                                                C=dictionary_values['LinearSVC.C']))
         
        if position == 44:
            estimator = ('LogisticRegression', LogisticRegression(penalty=dictionary_values['LogisticRegression.penalty'],
                                                                  C=dictionary_values['LogisticRegression.C'],
                                                                  dual=dictionary_values['LogisticRegression.dual'],
                                                                  solver=dictionary_values['LogisticRegression.solver']))      
        if position == 49:
            estimator = ('Binarizer', Binarizer(threshold=dictionary_values['Binarizer.threshold']))          
        if position == 51:
            estimator = ('FastICA', FastICA(tol=dictionary_values['FastICA.tol']))             
        if position == 53:
            estimator = ('FeatureAgglomeration', FeatureAgglomeration(linkage=dictionary_values['FeatureAgglomeration.linkage'],
                                                                      affinity=dictionary_values['FeatureAgglomeration.affinity']))           
        if position == 56:
            estimator = ('MaxAbsScaler', MaxAbsScaler())            
        if position == 57:
            estimator = ('MinMaxScaler', MinMaxScaler())             
        if position == 58:
            estimator = ('Normalizer', Normalizer(norm=dictionary_values['Normalizer.norm']))             
        if position == 60:
            estimator = ('Nystroem', Nystroem(kernel=dictionary_values['Nystroem.kernel'],
                                              gamma=dictionary_values['Nystroem.gamma'],
                                              n_components=dictionary_values['Nystroem.n_components']))            
        if position == 64:
            estimator = ('PCA', PCA(svd_solver=dictionary_values['PCA.svd_solver'],
                                    iterated_power=dictionary_values['PCA.iterated_power'])) 
        if position == 67:
            estimator = ('PolynomialFeatures', PolynomialFeatures(degree=dictionary_values['PolynomialFeatures.degree'],
                                                                  include_bias=dictionary_values['PolynomialFeatures.include_bias'],
                                                                  interaction_only=dictionary_values['PolynomialFeatures.interaction_only']))           
        if position == 71:
            estimator = ('RBFSampler', RBFSampler(gamma=dictionary_values['RBFSampler.gamma']))  
        if position == 73:
            estimator = ('RobustScaler', RobustScaler())  
        if position == 74:
            estimator = ('StandardScaler', StandardScaler())            
        if position == 75:
            estimator = ('SelectFwe', SelectFwe(alpha=dictionary_values['SelectFwe.alpha'],
                                                score_func=dictionary_values['SelectFwe.score_func']))           
        if position == 77:
            estimator = ('SelectPercentile', SelectPercentile(percentile=dictionary_values['SelectPercentile.percentile'],
                                                              score_func=dictionary_values['SelectPercentile.score_func']))            
        if position == 79:
            estimator = ('VarianceThreshold', VarianceThreshold(threshold=dictionary_values['VarianceThreshold.threshold']))               
        return estimator
   
newInstance = ValuesSearchSpace()
for i in export:
    result1 = newInstance.get_individuals(i)
    print(result1)