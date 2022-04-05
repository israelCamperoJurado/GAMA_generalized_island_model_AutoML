# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:59:53 2021

@author: 20210595
"""

from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

# Packages for pygmo
import os
import pickle
from shutil import rmtree
from numpy import genfromtxt
import numpy as np
import pygmo as pg
from pygmo import *
from gama.configuration.bounds_pygmo import (
    upperBound, 
    lowerBound, 
    vector_support,
    count_aux
    ) 
from gama.configuration.create_individuals import ValuesSearchSpace, IndividuoVector

import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.logging.evaluation_logger import EvaluationLogger
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_evaluator import AsyncEvaluator

log = logging.getLogger(__name__)
    
class SearchPygmo(BaseSearch):
    """ Perform asynchronous evolutionary optimization.
    Parameters
    ----------
    population_size: int, optional (default=50)
        Maximum number of individuals in the population at any time.
    max_n_evaluations: int, optional (default=None)
        If specified, only a maximum of `max_n_evaluations` individuals are evaluated.
        If None, the algorithm will be run until interrupted by the user or a timeout.
    restart_callback: Callable[[], bool], optional (default=None)
        Function which takes no arguments and returns True if search restart.
    """

    def __init__(
        self,
        population_size: Optional[int] = None,
        max_n_evaluations: Optional[int] = None,
        restart_callback: Optional[Callable[[], bool]] = None,
    ):
        super().__init__()
        # maps hyperparameter -> (set value, default)
        self._hyperparameters: Dict[str, Tuple[Any, Any]] = dict(
            population_size=(population_size, 50),
            restart_callback=(restart_callback, None),
            max_n_evaluations=(max_n_evaluations, None),
        )
        self.output = []

        def get_parent(evaluation, n) -> str:
            """ retrieves the nth parent if it exists, '' otherwise. """
            if len(evaluation.individual.meta.get("parents", [])) > n:
                return evaluation.individual.meta["parents"][n]
            return ""

        self.logger = partial(
            EvaluationLogger,
            extra_fields=dict(
                parent0=partial(get_parent, n=0),
                parent1=partial(get_parent, n=1),
                origin=lambda e: e.individual.meta.get("origin", "unknown"),
            ),
        )

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        self.output = pygmo_serach(
            operations, self.output, start_candidates, **self.hyperparameters
        ) 

   
def loss_function(ops: OperatorSet, ind1: Individual) -> Individual:
    with AsyncEvaluator() as async_:
        async_.submit(ops.evaluate, ind1)
        future = ops.wait_next(async_)
        if future.exception is None:
            individual_prototype = future.result.individual
        return individual_prototype

class AutoMLProblem:
    #save_whiout_evaluation = []
    contador = 0
    def __init__(self, ops, name="individual"):
        self.operator = ops
        self.output = []
        self.count = 0
        self.name = name
        self.name_previous = name
        self.old_loss = 1000
        self.new_loss = None
        
    # Define objectives
    def fitness(self, x):
        #path = path + '/individual10.pkl'
        #os.path.isfile(path)
        AutoMLProblem.contador += 1
        path_use = os.getcwd()
        path = path_use.replace(os.sep, '/')
        path = path + "/pickle_gama/" + self.name + ".pkl"
        instance_individual = ValuesSearchSpace(x)
        individual_from_x = instance_individual.get_individuals()
        if individual_from_x == None:
            print("El individuo era None")
            f1 = -1000
        else:
            try:
                individual_to_use = self._loss_function(self.operator, individual_from_x)
                print("Individual evaluated with PyGMO Search Multi-Archipelago 50 generations", individual_to_use)
                f1 = individual_to_use.fitness.values[0]
                if f1 == -np.inf:
                    f1 = -1000
                if -f1 < self.old_loss:
                    self.old_loss = -f1
                    print("The loss is lower than the previous one", self.old_loss)
                    self.output.append(individual_to_use)
                    
                    while(os.path.isfile(path)):
                        print("El camino es: ", path)
                        self.count += 1
                        self.name = self.name_previous + '%d' % self.count 
                        path_use = os.getcwd()
                        path = path_use.replace(os.sep, '/')
                        path = path + "/pickle_gama/" + self.name + ".pkl"
                        print("Nuevo camino es: ", path)
                            
                    with open(path, 'wb') as f:
                        pickle.dump(self.output, f)
            except:
                f1 = -1000
        return [-f1]
                
            
                        
            # if AutoMLProblem.contador > 1:
            #     try:
            #         with open(path, 'rb') as f:
            #             self.output = pickle.load(f)
            #         self.output.append(individual_to_use)
            #         print('len self.output', len(self.output))
            #         with open(path, 'wb') as f:
            #             pickle.dump(self.output, f)
            #     except:
            #         #self.output = []
            #         #self.output.append(individual_to_use)
            #         print("entré a la excepción de pickle")
            #         self.count += 1
            #         #self.name = 'individual%d' % self.count + ".pkl"
            #         self.name = self.name_previous + '%d' % self.count 
            #         with open(path, 'wb') as f:
            #             pickle.dump(self.output, f)
        return [-f1]
    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
    def _loss_function(self, ops: OperatorSet, ind1: Individual) -> Individual:
        #individual = ops.evaluate(ind1).individual
        print("Hi Pieter", **AsyncEvaluator.defaults)
        #help(ops.evaluate)
        result = ops.evaluate(ind1)
        #result = ops.evaluate(ind1, **AsyncEvaluator.defaults)
        return result.individual
    
    # Return function name
    def get_name(self):
        return "AutoMLProblem"
 
 
def pygmo_serach(
    ops: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    restart_callback: Optional[Callable[[], bool]] = None,
    max_n_evaluations: Optional[int] = None,
    population_size: int = 50,
    islands: int = 8,
    iters: int = 1,
    #iters: int = 10,
) -> List[Individual]:
    list_archipelagos = []
    #print(AsyncEvaluator.defaults)   
    current_population = output
    print("------------------------------------------------")
    print("Iterations new WITHOUT wait_check()", iters)
    
    #Create a folder to save the invididuals
    path_use = os.getcwd()
    path = path_use.replace(os.sep, '/')
    path = path + "/pickle_gama"
        
    try: 
        os.mkdir(path) 
    except: 
        rmtree(path)
        os.mkdir(path) 
    
    lista_aux = []
    f_vectors = []
    path_use = os.getcwd()
    path = path_use.replace(os.sep, '/')
    path = path + "/pickle_gama/" + "warm_start" + ".pkl"
    for individual in start_candidates:
        result = ops.evaluate(individual)
        new_ind = result.individual
        loss = new_ind.fitness.values[0]
        f_vectors.append(loss)
        lista_aux.append(new_ind)
        with open(path, 'wb') as f:
            pickle.dump(lista_aux, f)
        
    x_vectors = []
    for i in lista_aux:
        instance_individual_to_vectors = IndividuoVector()
        new_vector = instance_individual_to_vectors(i)
        x_vectors.append(new_vector)
        
 
    final_output = []
    x_of_island_champion = x_vectors.copy()
    print("Jalada ________________________________________________")
    for k in x_of_island_champion:
        try:
            final_instance = ValuesSearchSpace(k)
            individual_from_x = final_instance.get_individuals()
            result = ops.evaluate(individual_from_x)
            new_ind = result.individual
            final_output.append(new_ind)
        except:
            print("Ese individuo no será evaluado")

    current_population=final_output

        
    
    print("Longitud final", len(current_population))
    return current_population


    # import os
    # import pickle
    # path_use = os.getcwd()
    # path = path_use.replace(os.sep, '/')
    # path = path + "/list_successive_halving.pkl"  
    # lista = [1,1,1,1,1]
    # with open(path, 'wb') as f:
    #     pickle.dump(lista, f)    
    # # for root, dirs, files, in os.walk(path):
    # #     for file in files:
    # #         if file.endswith(".pkl"):
    # #             print(file)
    # #             # if file == "buscar.pkl":
    # #             #     os.remove(file)
                
    # # path = path + "/"+ "list_successive_halving.pkl"
    # # list_successive_halving = pickle.load(open(path, "rb"))