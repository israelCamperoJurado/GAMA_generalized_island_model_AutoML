# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:27:24 2021

@author: 20210595
"""


from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

# Packages for pygmo
import pickle
from numpy import genfromtxt
import numpy as np
import pygmo as pg
from pygmo import *
from gama.configuration.bounds_pygmo import (
    upperBound, 
    lowerBound, 
    vector_support
    ) 
from gama.configuration.create_individuals import ValuesSearchSpace

import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.logging.evaluation_logger import EvaluationLogger
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_evaluator import AsyncEvaluator
#from gama.utilities.generic.async_evaluator_pygmo import AsyncEvaluator

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

# nueva = 2
# variable = 'f%d' % nueva
# with open('output.txt', 'w') as variable:
#     print(variable)
#     variable.write('Hi there!')
    
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
    def __init__(self, ops):
        self.operator = ops
        self.output = []
        
    # Define objectives
    def fitness(self, x):
        instance_individual = ValuesSearchSpace(x)
        individual_from_x = instance_individual.get_individuals()
        individual_to_use = self._loss_function(self.operator, individual_from_x)
        self.output.append(individual_to_use)
        with open('parrot.pkl', 'wb') as f:
            pickle.dump(self.output, f)
        #AutoMLProblem.save_whiout_evaluation.append(individual_from_x)
        # self.output.append(individual_to_use)
        print("Individual evaluated with PyGMO Search Multi-Archipelago", individual_to_use)
        #AutoMLProblem.save_ind.append(individual_to_use)
        #self.output = AutoMLProblem.save_ind
        f1 = individual_to_use.fitness.values[0]
        if f1 == -np.inf:
            f1 = -1000
        #print(AutoMLProblem.save_ind)
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
    population_size: int = 5,
    islands: int = 5,
    iters: int = 5,
) -> List[Individual]:
    
    current_population = output
    
    # with AsyncEvaluator() as async_:
    #     for individual in start_candidates:
    #         async_.submit(ops.evaluate, individual)
    #         future = ops.wait_next(async_)
    #         if future.exception is None:
    #             individual_prototype = future.result.individual
    #             print("Individual start_candidates evaluated", individual_prototype)
    #             current_population.append(individual_prototype)
    #             print(individual_prototype.fitness.values[0], type)
    #             if individual_prototype.fitness.values[0] == -np.inf:
    #                 print("infinito")
    print(AsyncEvaluator.defaults)     
    print("START with pygmo")
    algo = pg.algorithm(pg.de(gen = iters))
    prob = pg.problem(AutoMLProblem(ops))        
    # The initial population
    pop = pg.population(prob)
    class_support = AutoMLProblem(ops)
    x_vectors = genfromtxt('x_to_save.csv', delimiter=',')
    f_vectors = genfromtxt('f_to_save.csv', delimiter=',')
    for i in range(len(x_vectors)):
        if f_vectors[i] == -np.inf:
            f_vectors[i] = -10000
        pop.push_back(x = x_vectors[i].tolist(), f = [-f_vectors[i]])
    archi = pg.archipelago(n=4, algo=algo, pop=pop)
    print("CREATION OF THE ARCHIPELAGO, IT WILL START THE EVOLUTION IN PARALLEL")
    print(archi) 
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    archi.evolve()
    archi.wait()
    archi.wait_check()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    print("IT JUST FINISH")
    print(archi)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    with open('parrot.pkl', 'rb') as f:
        mynewlist = pickle.load(f)
    # x_of_island_champion = archi.get_champions_x()
    # final_output = []
    # for i in x_of_island_champion:
    #     final_instance = ValuesSearchSpace(i)
    #     individual_from_x = final_instance.get_individuals()
    #     individual_to_use = loss_function(ops, individual_from_x)
    #     final_output.append(individual_to_use)
    # current_population = current_population + final_output
    print("All the individuals", mynewlist)
    print("Longitud final", len(mynewlist))
    current_population=mynewlist
    return current_population

    