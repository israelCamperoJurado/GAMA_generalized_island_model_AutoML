# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:30:12 2021

@author: 20210595
"""

from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

# Packages for pygmo
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
        print("Entré a la clase Pygmo Search 4")
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
        #print("self.output, aync_ea", self.output)


# def pygmo_serach(
#     ops: OperatorSet,
#     output: List[Individual],
#     start_candidates: List[Individual],
#     restart_callback: Optional[Callable[[], bool]] = None,
#     max_n_evaluations: Optional[int] = None,
#     population_size: int = 50,
# ) -> List[Individual]:
#     """ Perform asynchronous evolutionary optimization with given operators.

#     Parameters
#     ----------
#     ops: OperatorSet
#         Operator set with `evaluate`, `create`, `individual` and `eliminate` functions.
#     output: List[Individual]
#         A list which contains the set of best found individuals during search.
#     start_candidates: List[Individual]
#         A list with candidate individuals which should be used to start search from.
#     restart_callback: Callable[[], bool], optional (default=None)
#         Function which takes no arguments and returns True if search restart.
#     max_n_evaluations: int, optional (default=None)
#         If specified, only a maximum of `max_n_evaluations` individuals are evaluated.
#         If None, the algorithm will be run indefinitely.
#     population_size: int (default=50)
#         Maximum number of individuals in the population at any time.

#     Returns
#     -------
#     List[Individual]
#         The individuals currently in the population.
#     """
#     if max_n_evaluations is not None and max_n_evaluations <= 0:
#         raise ValueError(
#             f"n_evaluations must be non-negative or None, is {max_n_evaluations}."
#         )

#     max_pop_size = population_size

#     current_population = output
#     n_evaluated_individuals = 0

#     with AsyncEvaluator() as async_:
#         should_restart = True
#         while should_restart:
#             should_restart = False
#             current_population[:] = []
#             log.info("Starting EA with new population.")
#             for individual in start_candidates:
#                 async_.submit(ops.evaluate, individual)

#             while (max_n_evaluations is None) or (
#                 n_evaluated_individuals < max_n_evaluations
#             ):
#                 if n_evaluated_individuals < population_size:
#                     future = ops.wait_next(async_)
#                     if future.exception is None:
#                         individual = future.result.individual
#                         print("individuo evaluado", individual)
#                         current_population.append(individual)
#                         if len(current_population) > max_pop_size:
#                             to_remove = ops.eliminate(current_population, 1)
#                             current_population.remove(to_remove[0])
                        
#                 if len(current_population) > int(max_pop_size/2):
#                     # new_individual = ops.create(current_population, 1)[0]
#                     # async_.submit(ops.evaluate, new_individual)
#                     instance_support = AutoMLProblem(ops)
#                     try:
#                         current_population = main(ops)
#                         if len(instance_support.save_whiout_evaluation) > 1:
#                             new_individual = instance_support.save_whiout_evaluation[-1]
#                             async_.submit(ops.evaluate, new_individual)
#                     except:
#                         print("Entre en la excepcion y voy a imprimir la poblacion")
#                         current_population = current_population + instance_support.save_ind
#                         print('poblacion de excepcion:')
#                         for i in current_population:
#                             print(i)
#                     print("Hemos actualizado la poblaición actual, lista de individuos")

#                 should_restart = restart_callback is not None and restart_callback()
#                 n_evaluated_individuals += 1
#                 print('n_evaluated_individuals', n_evaluated_individuals)
#                 if should_restart:
#                     log.info("Restart criterion met. Creating new random population.")
#                     start_candidates = [ops.individual() for _ in range(max_pop_size)]
#                     break

#     return current_population

    
def loss_function(ops: OperatorSet, ind1: Individual) -> Individual:
    with AsyncEvaluator() as async_:
        async_.submit(ops.evaluate, ind1)
        future = ops.wait_next(async_)
        if future.exception is None:
            individual_prototype = future.result.individual
        return individual_prototype
    
class AutoMLProblem:
    save_ind = []
    save_whiout_evaluation = []
    def __init__(self, ops):
        self.operator = ops
        self.output = []
    # Define objectives
    def fitness(self, x):
        instance_individual = ValuesSearchSpace(x)
        individual_from_x = instance_individual.get_individuals()
        individual_to_use = loss_function(self.operator, individual_from_x)
        AutoMLProblem.save_whiout_evaluation.append(individual_to_use)
        # self.output.append(individual_to_use)
        print("Individuo evaluado en PyGMO Search ", individual_to_use)
        AutoMLProblem.save_ind.append(individual_to_use)
        self.output = AutoMLProblem.save_ind
        f1 = individual_to_use.fitness.values[0]
        return [f1]
    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
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
    iters: int = 40,
) -> List[Individual]:
    
    current_population = output
    n_evaluated_individuals = 0
    
    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            async_.submit(ops.evaluate, individual)
            future = ops.wait_next(async_)
            if future.exception is None:
                individual_prototype = future.result.individual
                print("Individual start_candidates evaluated", individual_prototype)
                current_population.append(individual_prototype)
            
        try:
            class_support = AutoMLProblem(ops)
            print("Iniciar con pygmo")
            # algo = pg.algorithm(pg.de(gen=iters))
            # prob = pg.problem(AutoMLProblem(ops))
            # archi = pg.archipelago(n=islands, algo=algo, prob=prob, pop_size=10)
            # archi.evolve() 
            # archi.wait()
            # # fitness_of_island_champion = archi.get_champions_f()
            # x_of_island_champion = archi.get_champions_x()
            algo = pg.algorithm(pg.de(gen = iters))
            prob = pg.problem(AutoMLProblem(ops))
            pop = pg.population(prob, population_size)
            pop = algo.evolve(pop) 
            final_output = []
            #for i in x_of_island_champion:
            for i in pop:
                final_instance = ValuesSearchSpace(i)
                individual_from_x = final_instance.get_individuals()
                individual_to_use = loss_function(ops, individual_from_x)
                final_output.append(individual_to_use)
            print('final_output, ya terminé un pygmo')
            for i in final_output:
                print(i)
            current_population = final_output
        except:
            print("Entre en la excepcion y voy a imprimir la poblacion")
            current_population = current_population + class_support.save_ind
            print('poblacion de excepcion:')
            for i in current_population:
                print(i)
            print("Hemos actualizado la poblaición actual, lista de individuos")
    return current_population

# def main(operator, iters=4, pop_size=10):
#     algo = pg.algorithm(pg.de(gen = iters))
#     prob = pg.problem(AutoMLProblem(operator))
#     pop = pg.population(prob, pop_size)
#     pop = algo.evolve(pop) 
#     final_output = []
#     for i in pop:
#         final_instance = ValuesSearchSpace(i)
#         individual_from_x = final_instance.get_individuals()
#         individual_to_use = loss_function(operator, individual_from_x)
#         final_output.append(individual_to_use)
#     print('final_output, ya terminé un pygmo', final_output)
#     return final_output