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


class RandomSearch(BaseSearch):
    """ Perform random search over all possible pipelines. """
    #print("Entre a la clase random")
    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        #print("Entré a search de Random Search")
        #print("start_candidates, Random Serach")
        #print("operations Random Search", operations, " and type", type(operations))
        random_search(operations, self.output, start_candidates)


def random_search(
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
    #print("operations in Random Search", operations)
    _check_base_search_hyperparameters(operations, output, start_candidates)
    #print(AsyncEvaluator.defaults)
    with AsyncEvaluator() as async_:
        for individual in start_candidates:
            async_.submit(operations.evaluate, individual)

        while (max_evaluations is None) or (len(output) < max_evaluations):
            future = operations.wait_next(async_)
            if future.result is not None:
                output.append(future.result.individual)
                #print("Len output random serach", len(output))
                #print("Imprimir primer individuo de random search", output[0])
            async_.submit(operations.evaluate, operations.individual())
    return output
