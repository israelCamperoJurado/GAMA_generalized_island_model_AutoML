# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:28:28 2021

@author: 20210595
"""
from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal
# import multiprocessing
# from multiprocessing import Process, freeze_support


# freeze_support()

# Packages for pygmo
import os
import pickle
import uuid
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
        
        path_use = os.getcwd()
        path = path_use.replace(os.sep, '/')
        name_folder = "pickle_gama_" + str(uuid.uuid4())
        path = path + "/" + name_folder
        self.path = path
        print("search method", self.path)
        
        # self.manager = multiprocessing.Manager()
        # self.final_list = self.manager.list()

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
            operations, self.output, start_candidates, self.path, **self.hyperparameters
        ) 


class AutoMLProblem(object):
    #save_whiout_evaluation = []
    # contador = 0
    def __init__(self, ops, folder_name):
        self.operator = ops
        self.output = []
        self.name = "individual"
        self.name_previous = "individual"
        self.old_loss = 1000
        self.new_loss = None
        self.folder_name = folder_name
        
    # Define objectives
    def fitness(self, x):
        #path = path + '/individual10.pkl'
        #os.path.isfile(path)
        # AutoMLProblem.contador += 1
        # path_use = os.getcwd()
        # path = path_use.replace(os.sep, '/')
        # path = path + "/pickle_gama/" + self.name + ".pkl"
        # path_ind = folder_name + "/" + self.name + ".pkl"
        instance_individual = ValuesSearchSpace(x)
        individual_from_x = instance_individual.get_individuals()
        if individual_from_x == None:
            f1 = -1000
        else:
            try:
                if individual_from_x != None:
                    individual_to_use = self._loss_function(self.operator, individual_from_x)
                    f1 = individual_to_use.fitness.values[0]
                    if f1 == -np.inf:
                        f1 = -1000
                    list_save_ind = [individual_to_use]
                    self.name = self.name_previous + str(uuid.uuid4())
                    path_ind = self.folder_name + "/" + self.name + ".pkl"
                    with open(path_ind, 'wb') as f:
                        pickle.dump(list_save_ind, f)
                else:
                    f1 = -1000
            except:
                f1 = -1000
        return [-f1]
    
    

# class AutoMLProblem(object):
#     #save_whiout_evaluation = []
#     # contador = 0
#     def __init__(self, ops, folder_name):
#         self.operator = ops
#         self.output = []
#         self.name = "individual"
#         self.name_previous = "individual"
#         self.old_loss = 1000
#         self.new_loss = None
#         self.folder_name = folder_name
        
#     # Define objectives
#     def fitness(self, x):
#         #path = path + '/individual10.pkl'
#         #os.path.isfile(path)
#         # AutoMLProblem.contador += 1
#         # path_use = os.getcwd()
#         # path = path_use.replace(os.sep, '/')
#         # path = path + "/pickle_gama/" + self.name + ".pkl"
#         # path_ind = folder_name + "/" + self.name + ".pkl"
#         instance_individual = ValuesSearchSpace(x)
#         individual_from_x = instance_individual.get_individuals()
#         if individual_from_x == None:
#             f1 = -1000
#         else:
#             try:
#                 if individual_from_x != None:
#                     individual_to_use = self._loss_function(self.operator, individual_from_x)
#                     # print("Individual evaluated with PyGMO Search Multi-Archipelago 50 generations", individual_to_use)
#                     f1 = individual_to_use.fitness.values[0]
#                     if f1 == -np.inf:
#                         f1 = -1000
#                     if -f1 < self.old_loss:
#                         self.old_loss = -f1
#                         # print("The loss is lower than the previous one", self.old_loss)
#                         self.output.append(individual_to_use)
                        
#                         # print("El camino es: ", path)
#                         self.name = self.name_previous + str(uuid.uuid4())
#                         # path_use = os.getcwd()
#                         # path = path_use.replace(os.sep, '/')
#                         # path = path + "/pickle_gama/" + self.name + ".pkl"
#                         path_ind = self.folder_name + "/" + self.name + ".pkl"
#                         # print("Nuevo camino es: ", path)
                                
#                         with open(path_ind, 'wb') as f:
#                             pickle.dump(self.output, f)
#                 else:
#                     # print("Voy a imprimir el individuo", individual_to_use)
#                     f1 = -1000
#             except:
#                 f1 = -1000
#         return [-f1]
                

    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
    def _loss_function(self, ops: OperatorSet, ind1: Individual) -> Individual:
        #individual = ops.evaluate(ind1).individual
        # print("Hi Pieter", **AsyncEvaluator.defaults)
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
    path,
    restart_callback: Optional[Callable[[], bool]] = None,
    max_n_evaluations: Optional[int] = None,
    population_size: int = 50,
    islands: int = 8,
    iters: int = 50,
    #iters: int = 10,
) -> List[Individual]:
    
    #Create a folder to save the invididuals
    # path_use = os.getcwd()
    # path = path_use.replace(os.sep, '/')
    # name_folder = "pickle_gama_" + str(uuid.uuid4())
    # path = path + "/" + name_folder
    path = path
    os.makedirs(path, exist_ok=True)
    
    
    path_warm = path + "/" + "warm_start" + ".pkl"
    f_vectors = []
    for individual in start_candidates:
        result = ops.evaluate(individual)
        new_ind = result.individual
        loss = new_ind.fitness.values[0]
        f_vectors.append(loss)
        output.append(new_ind)
        with open(path_warm, 'wb') as f:
            pickle.dump(output, f)
            
    x_vectors = []
    for i in output:
        instance_individual_to_vectors = IndividuoVector()
        new_vector = instance_individual_to_vectors(i)
        x_vectors.append(new_vector)
            
    print("Ya convertí el warm-start en vectores, new method")
                  
    print("START with pygmo")    
    # algo = pg.algorithm(pg.de(gen = iters))
    prob = pg.problem(AutoMLProblem(ops, path))    
    # The initial population
    pop = pg.population(prob)
    for i in range(len(x_vectors)):
        if f_vectors[i] == -np.inf:
            f_vectors[i] = -10000
        pop.push_back(x = x_vectors[i], f = [-f_vectors[i]])
            
    # Changes from here
    r_policy = pg.r_policy(pg.fair_replace(rate=0.5)) # Share 50% of the individulas en each island
    s_policy = pg.s_policy(udsp=pg.select_best())
    archi = pg.archipelago(r_pol=r_policy, s_pol=s_policy, t=pg.topology(pg.fully_connected()))
    broadcast = pg.migration_type(1) # 1 = Broadcast type
    archi.set_migration_type(broadcast)
    # To here
        
    # archi = pg.archipelago(t=pg.topology(pg.ring()))
    # archi = pg.archipelago(t=pg.topology(pg.fully_connected()))
    # archi = pg.archipelago(t=pg.topology(pg.free_form()))
    # archi = pg.archipelago() # unconnected topology
    # archi = pg.archipelago(t=pg.topology(pg.base_bgl_topology())) # this doesn't work, is only the base of a topology
    isl1 = pg.island(algo = pg.algorithm(pg.de(gen = iters)), pop=pop)
    isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = iters)), pop=pop)
    isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = iters)), pop=pop)
    isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = iters)), pop=pop)
    isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = iters)), pop=pop)
    isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = iters)), pop=pop)
    isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = iters)), pop=pop)
    isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = iters)), pop=pop)
    isls = [isl1, isl2, isl3, isl4, isl5, isl6, isl7, isl8]

    for isl in isls:
        archi.push_back(isl)
    print("Acabo de CREAR EL ARCHIPELAGO, EMPEZARÉ A EVOLUCIONAR EN PARALELO")
            
    #archi = pg.archipelago(n=islands, algo=algo, pop=pop, t=pg.topology(pg.ring()))
    print("CREATION OF THE ARCHIPELAGO, IT WILL START THE EVOLUTION IN PARALLEL")
    print(archi) 
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    archi.evolve()
    # archi.wait()
    archi.wait_check()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    print("IT JUST FINISH")
    print(archi)
    print("Let's start with the iterative process")
    print("len archi.get_champions_f()", len(archi.get_champions_f()))
    print("len archi.get_champions_x()[0]", len(archi.get_champions_x()[0]))
    print("len archi.get_champions_x()", len(archi.get_champions_x()))
    
    
    # final_output = []
    x_of_island_champion = archi.get_champions_x()
    print("El archipelago tiene ", len(x_of_island_champion), " nuevos individuos")
    for k in x_of_island_champion:
        final_instance = ValuesSearchSpace(k)
        individual_from_x = final_instance.get_individuals()
        result = ops.evaluate(individual_from_x)
        new_ind = result.individual
        output.append(new_ind)
            
    return output
