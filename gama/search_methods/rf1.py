# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:48:33 2021

@author: 20210595
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 17:20:29 2021

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
terminal6 = Terminal(True, 'RandomForestClassifier.bootstrap', 'RandomForestClassifier.bootstrap')

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


import logging
from typing import List, Optional

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.search_methods.base_search import (
    BaseSearch,
    _check_base_search_hyperparameters,
)
from gama.utilities.generic.async_evaluator import AsyncEvaluator

log = logging.getLogger(__name__)


class RandomForestTry(BaseSearch):
    """ Perform asynchronous evolutionary optimization""" 
    print("RandomForestTry")
    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        randomForestSearch(operations, self.output, start_candidates)
        print("Entré a RandomForestTry search")

def randomForestSearch(
    operations: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    max_evaluations: Optional[int] = None,
) -> List[Individual]:
    """ Perform random search over all possible pipelines.

    Parameters
    ----------
    operations: OperatorSet
        An operator set with `evaluate` and `individual` functions.
    output: List[Individual]
        A list which contains the found individuals during search.
    start_candidates: List[Individual]
        A list with candidate individuals to evaluate first.
    max_evaluations: int, optional (default=None)
        If specified, only a maximum of `max_evaluations` individuals are evaluated.
        If None, the algorithm will be run indefinitely.

    Returns
    -------
    List[Individual]
        All evaluated individuals.
    """
    #print('outputs random search', output)
    #print("Len start candidates", len(start_candidates))
    #print('outputs random search', start_candidates)
    print("operations in RandomForestTry", operations)
    start_candidates=[ind] + start_candidates
    _check_base_search_hyperparameters(operations, output, start_candidates)

    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            # print("individuo antes del async_.submit en RF", individual)
            async_.submit(operations.evaluate, individual)
            # print("individuo despues del async_.submit en RF", individual)
        contador = 0
        while (max_evaluations is None) or (len(output) < max_evaluations):
            future = operations.wait_next(async_)
            #print("operations.wait_next(async_)", type(operations.wait_next(async_)))
            if future.result is not None:
                output.append(future.result.individual) # Aqui pasa la mágia
                if contador == 0:
                    print("Imprimir primer el bosque aleatorio evaluado", output[0])
                contador += 1
                # print("Len output random serach", len(output))
                # print("output, random_search")
                # print("Imprimir primer individuo de random search", output[0])
                # print("Imprimir segundo individuo de random search", output[1])
            async_.submit(operations.evaluate, operations.individual())
    return output
