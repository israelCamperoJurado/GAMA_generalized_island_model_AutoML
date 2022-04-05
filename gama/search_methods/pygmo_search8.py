# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:04:29 2021

@author: 20210595
"""


# Packages for pygmo
import pygmo as pg
from pygmo import *
from gama.configuration.bounds_pygmo import (
    upperBound, 
    lowerBound
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
    save_ind = []
    save_whiout_evaluation = []
    contador = 0
    def __init__(self, ops):
        self.operator = ops
        self.output = []

    def fitness(self, x):
        AutoMLProblem.contador += 1
        print("Número de modelo evaluado", AutoMLProblem.contador)
        with AsyncEvaluator() as async_:
            instance_individual = ValuesSearchSpace(x)
            individual_from_x = instance_individual.get_individuals()
            async_.submit(self.operator.evaluate, individual_from_x)
            AutoMLProblem.save_whiout_evaluation.append(individual_from_x)
            future = self.operator.wait_next(async_)
            if future.exception is None:
                individual_prototype = future.result.individual
                print("Individuo evaluado en PyGMO Search DE ", individual_prototype)
                AutoMLProblem.save_ind.append(individual_prototype)
            self.output = AutoMLProblem.save_ind
            f1 = future.result.score[0]
            return [f1]
    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
    # Return function name
    def get_name(self):
        return "AutoMLProblem"
    
    def _loss_function(self, ops: OperatorSet, ind1: Individual) -> Individual:
        with AsyncEvaluator() as async_:
            async_.submit(ops.evaluate, ind1)
            future = ops.wait_next(async_)
            if future.exception is None:
                individual_prototype = future.result.individual
            return individual_prototype
 
 
def pygmo_serach(
    ops: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    restart_callback: Optional[Callable[[], bool]] = None,
    max_n_evaluations: Optional[int] = None,
    population_size: int = 50,
    islands: int = 8,
    iters: int = 50,
) -> List[Individual]:
    
    current_population = output
    n_evaluated_individuals = 0
    
    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            async_.submit(ops.evaluate, individual)
            future = ops.wait_next(async_)
            if future.exception is None:
                individual_prototype = future.result.individual
                current_population.append(individual_prototype)
                
        print("Iniciar con pygmo")
        algo = pg.algorithm(pg.de(gen = 5))
        prob = pg.problem(AutoMLProblem(ops))
        pop = pg.population(prob, 40)
        pop = algo.evolve(pop)
        final_pop = pop.get_x()
        final_output = []
        for i in final_pop:
            final_instance = ValuesSearchSpace(i)
            individual_from_x = final_instance.get_individuals()
            individual_to_use = loss_function(ops, individual_from_x)
            final_output.append(individual_to_use)
        print('final_output, ya terminé un pygmo')
        for i in final_output:
            print(i)
        current_population = current_population + final_output

    return current_population
