# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:51:27 2021

@author: 20210595
"""

from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

# Packages for pygmo
import pygmo as pg
import numpy as np
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
from gama.utilities.generic.async_evaluator_pygmo import AsyncEvaluator

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
    
def loss_function(ops: OperatorSet, ind1: Individual, identificador: int) -> Individual:
    new_name= identificador
    with AsyncEvaluator() as new_name:
        new_name.submit(ops.evaluate, ind1)
        future = ops.wait_next(new_name)
        if future.exception is None:
            individual_prototype = future.result.individual
        return individual_prototype
    
class AutoMLProblem:
    save_ind = []
    save_whiout_evaluation = []
    contador = 0
    def __init__(self, ops):
        self.operator = ops
        self.output = []
    # Define objectives
    def fitness(self, x):
        instance_individual = ValuesSearchSpace(x)
        individual_from_x = instance_individual.get_individuals()
        individual_to_use = self._loss_function(self.operator, individual_from_x)
        AutoMLProblem.save_whiout_evaluation.append(individual_to_use)
        # self.output.append(individual_to_use)
        print("Individuo evaluado en PyGMO Search DE Archipelago", individual_to_use)
        AutoMLProblem.save_ind.append(individual_to_use)
        self.output = AutoMLProblem.save_ind
        f1 = individual_to_use.fitness.values[0]
        if f1 == -np.inf:
            f1 = 10000
        print("f1 to return", f1)
        return [f1]
    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
    def _loss_function(self, ops: OperatorSet, ind1: Individual) -> Individual:
        AutoMLProblem.contador += 1
        print('Numero de modelos evaluados', AutoMLProblem.contador)
        new_name = 'async_%d' % AutoMLProblem.contador
        with AsyncEvaluator() as new_name:
            new_name.submit(ops.evaluate, ind1)
            future = ops.wait_next(new_name)
            if future.exception is None:
                individual_prototype = future.result.individual
            return individual_prototype
    
    # Return function name
    def get_name(self):
        return "AutoMLProblem"
 
 
def pygmo_serach(
    ops: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    restart_callback: Optional[Callable[[], bool]] = None,
    max_n_evaluations: Optional[int] = None,
    population_size: int = 10,
    islands: int = 5,
    iters: int = 1000,
) -> List[Individual]:
    
    current_population = output
    
    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            async_.submit(ops.evaluate, individual)
            future = ops.wait_next(async_)
            if future.exception is None:
                individual_prototype = future.result.individual
                print("Individual start_candidates evaluated", individual_prototype)
                current_population.append(individual_prototype)
            
        print("Iniciar con pygmo")
        algo = pg.algorithm(pg.de(gen=iters))
        prob = pg.problem(AutoMLProblem(ops))        
        archi = pg.archipelago(t=pg.topology(pg.ring()))
        #isls = [pg.island(algo=algo, prob=prob, size=population_size, udi=pg.thread_island()) for i in range(islands)]
        # isl1 = pg.island(algo = pg.algorithm(pg.de(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        # isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = iters)), prob=prob, size=population_size, udi=pg.thread_island())
        isl1 = pg.island(algo = pg.algorithm(pg.de(gen = iters)), prob=prob, size=population_size)
        isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = iters)), prob=prob, size=population_size)
        isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = iters)), prob=prob, size=population_size)
        isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = iters)), prob=prob, size=population_size)
        isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = iters)), prob=prob, size=population_size)
        isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = iters)), prob=prob, size=population_size)
        isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = iters)), prob=prob, size=population_size)
        isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = iters)), prob=prob, size=population_size)
        isls = [isl1, isl2, isl3, isl4, isl5, isl6, isl7, isl8]
        for isl in isls:
            archi.push_back(isl)
        print("Acabo de CREAR EL ARCHIPELAGO, EMPEZARÉ A EVOLUCIONAR EN PARALELO")
        print(archi) 
        for i in range(500): 
            archi.evolve(1)
            archi.wait()
            #archi.wait_check()
        archi.get_champions_f() 
        print(archi.get_champions_f()) 
        print("YA TERMINÉ DE ESPERAR")
        print(archi)
        x_of_island_champion = archi.get_champions_x()
        final_output = []
        for i in x_of_island_champion:
            final_instance = ValuesSearchSpace(i)
            individual_from_x = final_instance.get_individuals()
            individual_to_use = loss_function(ops, individual_from_x)
            final_output.append(individual_to_use)
        current_population = current_population + final_output

    return current_population

