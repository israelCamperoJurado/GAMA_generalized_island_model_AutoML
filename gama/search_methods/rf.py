# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 17:20:29 2021

@author: 20210595
"""

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
    _check_base_search_hyperparameters(operations, output, start_candidates)

    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            # print("individuo antes del async_.submit en RF", individual)
            async_.submit(operations.evaluate, individual)
            # print("individuo despues del async_.submit en RF", individual)

        while (max_evaluations is None) or (len(output) < max_evaluations):
            future = operations.wait_next(async_)
            #print("operations.wait_next(async_)", type(operations.wait_next(async_)))
            if future.result is not None:
                output.append(future.result.individual) # Aqui pasa la mágia
                # print("Len output random serach", len(output))
                # print("output, random_search")
                # print("Imprimir primer individuo de random search", output[0])
                # print("Imprimir segundo individuo de random search", output[1])
            async_.submit(operations.evaluate, operations.individual())
    return output
