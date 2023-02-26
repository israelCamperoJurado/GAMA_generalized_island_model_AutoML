# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:28:28 2021

@author: 20210595
"""
from gama.search_methods.topologies_creation import obtain_topology

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
import stopit  # Is this the problem?

import logging
from functools import partial
from typing import Optional, Any, Tuple, Dict, List, Callable, Union

import pandas as pd
import math

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
            reduction_factor: Optional[int] = None,
            minimum_resource: Optional[Tuple[int, float]] = None,
            maximum_resource: Optional[Tuple[int, float]] = None,
            minimum_early_stopping_rate: Optional[int] = None,
    ):
        super().__init__()
        # maps hyperparameter -> (set value, default)
        self._hyperparameters: Dict[str, Tuple[Any, Any]] = dict(
            population_size=(population_size, 50),
            restart_callback=(restart_callback, None),
            max_n_evaluations=(max_n_evaluations, None),
            reduction_factor=(reduction_factor, 3),
            minimum_resource=(minimum_resource, 0.125),
            maximum_resource=(maximum_resource, 1.0),
            minimum_early_stopping_rate=(minimum_early_stopping_rate, 0),
        )
        self.output = []

        path_use = os.getcwd()
        path = path_use.replace(os.sep, '/')
        name_folder = "pickle_gama_" + str(uuid.uuid4())
        path = path + "/" + name_folder
        self.path = path

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
        set_max, default = self._hyperparameters["maximum_resource"]
        if set_max is not None and len(y) < set_max:
            # todo: take into account the evaluation procedure as well.
            logging.warning(
                f"`maximum_resource` was set to {set_max}, but the dataset only"
                f"contains {len(y)} samples. Reverting to default (1.0) instead."
            )
            self._hyperparameters["maximum_resource"] = (None, default)

    def search(self, operations: OperatorSet, start_candidates: List[Individual], time_limit: float):
        self.output = pygmo_serach(
            operations, self.output, start_candidates, self.path, time_limit, **self.hyperparameters
        )


def top(path=None): # Function for multiobjective and next rung
    lista_top = []
    for root, dirs, files, in os.walk(path):
        for file in files:
            if file.endswith(".pkl"):
                if (file != "warm_start"):
                    new_f_path = path + "/" + file
                    try:
                        new_lista = pickle.load(open(new_f_path, "rb"))
                    except:
                        new_lista = []
                    lista_top = lista_top + new_lista
    lista_ind_numpy = np.array(lista_top)
    f_vectors = [[new_individual.fitness.values[0], len(new_individual.pipeline)] for new_individual in lista_top] # We are including f1 and f2 here
    for i in range(len(f_vectors)):
        if f_vectors[i][0] == -np.inf: # [0] because it is only the loss function
            f_vectors[i][0] = -10000
    f_vectors = np.array(f_vectors)
    indices_best = f_vectors[:,0].argsort()[::-1]
    indices_best = indices_best.tolist()
    lista_to_return = lista_ind_numpy[indices_best]
    lista_to_return = lista_to_return.tolist()
    return lista_to_return # We are going to return the elements sorted (better to worse), we need to split them by islands

class AutoMLProblem(object):
    def __init__(self, ops, folder_name, rung, max_rung):
        self.operator = ops
        self.output = []
        self.name = "individual"
        self.name_previous = "individual"
        self.old_loss = 1000
        self.new_loss = None
        self.folder_name = folder_name
        self.rung = rung
        self.max_rung = max_rung

    # Define objectives
    def fitness(self, x):
        #print("self.rung: ", self.rung)
        instance_individual = ValuesSearchSpace(x)
        try:
            individual_from_x = instance_individual.get_individuals()
        except:
            f1 = -1000
            f2 = 10
            return [-f1, f2]
        if individual_from_x == None:
            f1 = -1000
        else:
            if individual_from_x != None:
                individual_to_use = self._loss_function(self.operator, individual_from_x)
                f1 = individual_to_use.fitness.values[0]
                f2 = len(individual_to_use.pipeline)
                if f1 == -np.inf:
                    f1 = -1000
                list_save_ind = [individual_to_use]
                self.name = self.name_previous + str(uuid.uuid4())
                path_ind = self.folder_name + "/" + self.name + ".pkl"
                with open(path_ind, 'wb') as f:
                    pickle.dump(list_save_ind, f)
            else:
                f1 = -1000
        return [-f1, f2] # -f1 because is neg_log_loss (higher better) and f2 because is len(pipeline) (lower better), In pagmo minimization is always assumed

    # Return number of objectives
    def get_nobj(self):
        return 2

    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)

    def _loss_function(self, ops: OperatorSet, ind1: Individual) -> Individual:
        rung = self.rung
        result = evaluate_on_rung(
            ind1, self.rung, self.max_rung, ops.evaluate
        )
        return result.individual

    # Return function name
    def get_name(self):
        return "AutoMLProblem"


def pygmo_serach(
        ops: OperatorSet,
        output: List[Individual],
        start_candidates: List[Individual],
        path,
        time_limit,
        reduction_factor: int = 2,
        minimum_resource: Union[int, float] = 0.125,
        maximum_resource: Union[int, float] = 1.0,
        minimum_early_stopping_rate: int = 0,
        max_full_evaluations: Optional[int] = None,
        restart_callback: Optional[Callable[[], bool]] = None,
        max_n_evaluations: Optional[int] = None,
        population_size: int = 50,
        islands: int = 8,
        iters: int = 50, # Previously was 10000
) -> List[Individual]:
    #start_candidates = [start_candidates[i] for i in range(40)]
    if not isinstance(minimum_resource, type(maximum_resource)):
        raise ValueError("Currently minimum and maximum resource must same type.")

    path = path
    os.makedirs(path, exist_ok=True)
    max_rung = math.ceil(
        math.log(maximum_resource / minimum_resource, reduction_factor)
    )
    rungs = range(minimum_early_stopping_rate, max_rung + 1)
    rung_resources = {
        rung: min(minimum_resource * (reduction_factor ** rung), maximum_resource)
        for rung in rungs
    }
    rung_individuals: Dict[int, List[Tuple[float, Individual]]] = {
        rung: [] for rung in reversed(rungs)
    }

    promoted_individuals: Dict[int, List[Individual]] = {
        rung: [] for rung in reversed(rungs)
    }
    list_to_save_archipelagos = []
    path_warm = path + "/" + "warm_start" + ".pkl"
    f_vectors = []
    for individual in start_candidates:
        result = ops.evaluate(individual)
        new_ind = result.individual
        loss = new_ind.fitness.values[0]
        length_pipeline = len(new_ind.pipeline)
        f_vectors.append([loss, length_pipeline])
        output.append(new_ind)
        with open(path_warm, 'wb') as f:
            pickle.dump(output, f)

    x_vectors = []
    for i in output:
        instance_individual_to_vectors = IndividuoVector()
        new_vector = instance_individual_to_vectors(i)
        x_vectors.append(new_vector)

    new_time_limit = (int(time_limit / len(rungs)))
    print("new_time_limit", new_time_limit)
    for rung in rungs:
        if rung == 0:
            print("Number or rung: ", rung)
            #with stopit.ThreadingTimeout(new_time_limit) as pyg_time:
            #with stopit.ThreadingTimeout(new_time_limit):
            ops.evaluate = partial(ops.evaluate, subsample=rung_resources[rung])
            prob = pg.problem(AutoMLProblem(ops, path, rung, max_rung))
            # minimum_number_of_islands = 16
            minimum_number_of_islands = 8 
            minimum_number_individuals_per_island = 8 # We want 8 individuals per island (X islands)
            number_population = int(len(x_vectors)/minimum_number_individuals_per_island)
            population_list = [pg.population(prob) for i in range(number_population)]
            isls = []
            for subpop in range(number_population):
                subpopulation_x = np.array(x_vectors)[subpop*minimum_number_individuals_per_island:(subpop+1)*minimum_number_individuals_per_island]
                subpopulation_f = np.array(f_vectors)[subpop*minimum_number_individuals_per_island:(subpop+1)*minimum_number_individuals_per_island]
                small_population_surrogate_x = subpopulation_x.copy()
                small_population_surrogate_f = subpopulation_f.copy()
                for i in range(len(subpopulation_x)):
                    if subpopulation_f[i][0] == -np.inf:
                        subpopulation_f[i][0] = -10000
                    population_list[subpop].push_back(x=subpopulation_x[i], f=[-subpopulation_f[i][0], subpopulation_f[i][1]])
                this_island = pg.island(algo=pg.algorithm(pg.nsga2(gen=iters)), pop=population_list[subpop])
                isls.append(this_island)
            # # The initial population
            #pop = pg.population(prob)
            #for i in range(len(x_vectors)):
            #    if f_vectors[i][0] == -np.inf:
            #        f_vectors[i][0] = -10000
            #    pop.push_back(x=x_vectors[i], f=[-f_vectors[i][0], f_vectors[i][1]])

            # Changes from here
            r_policy = pg.r_policy(pg.fair_replace(rate=0.5))  # Share 50% of the individulas en each island
            s_policy = pg.s_policy(udsp=pg.select_best())
            archi = pg.archipelago(r_pol=r_policy, s_pol=s_policy)
            #isl1 = pg.island(algo=pg.algorithm(pg.de(gen=iters)), pop=pop)
            #isl1 = pg.island(algo=pg.algorithm(pg.nsga2(gen=iters)), pop=pop)
            #isls = [isl1 for _ in range(16)]
            for isl in isls:
                archi.push_back(isl)
            # G = obtain_topology(name='complete_graph', nodes=int(len(archi)))
            # G = obtain_topology(name='circular_ladder_graph', nodes=int(len(archi) / 2))
            # G = obtain_topology(name = 'balanced_tree', nodes=int(len(archi)-1), h = 1)
            # G = obtain_topology(name = 'wheel', nodes=int(len(archi)))
            # G = obtain_topology(name='ladder_graph', nodes=int(len(archi) / 2))
            # G = obtain_topology(name = 'grid_graph', dim = (4, 4))
            # G = obtain_topology(name = 'grid_graph', dim = (4, 2, 2))
            # G = obtain_topology(name = 'hypercube_graph', nodes = 4)
            # G = obtain_topology(name = 'watts_strogatz_graph', nodes = int(len(archi)), k=2, p=0.5)
            # G = obtain_topology(name='cycle_graph', nodes=int(len(archi)))
            
            # # Small number of nodes (only 8 islands)
            G = obtain_topology(name = 'hypercube_graph', nodes = 3)
            # G = obtain_topology(name = 'grid_graph', dim = (4, 2))
            # G = obtain_topology(name = 'grid_graph', dim = (2, 2, 2))            
            
            this_topology = pg.free_form(G)
            # this_topology = pg.topology(pg.ring(len(archi)))
            archi.set_topology(this_topology)
            # print(archi.get_champions_f())
            archi.evolve()
            # archi.wait()
            archi.wait_check()
            #archi.get_champions_f()

            x_of_island_champion =[]
            for isl in archi:
                this_population = isl.get_population()
                ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(this_population.get_f())
                best_ind = pg.select_best_N_mo(points=this_population.get_f(), N=10)
                x_of_island_champion += list(this_population.get_x()[best_ind])

            #x_of_island_champion = archi.get_champions_x()
            for k in x_of_island_champion:
                final_instance = ValuesSearchSpace(k)
                individual_from_x = final_instance.get_individuals()
                result = ops.evaluate(individual_from_x)
                new_ind = result.individual
                # output.append(new_ind)
            list_to_save_archipelagos.append(archi)
        # if rung>0 and rung<max_rung:
        if rung > 0:
            if rung == max_rung: iters=200
            #with stopit.ThreadingTimeout(new_time_limit) as pyg_time_highest:
            #with stopit.ThreadingTimeout(new_time_limit):
            print("Number or rung: ", rung)
            # Let's start retrieve previous individuals from rung 1
            list_aux_save_individuals = []
            path_new_rung = path + "/" + "rung_iterations" + ".pkl"
            list_aux_save_individuals = list_aux_save_individuals + top(path=path)
            if rung == 1:
                number_of_configurations = len(list_aux_save_individuals)-1
                print("Number of individuals total for the next rungs is: ", number_of_configurations)

            # Let's convert the previous individuals into vectors
            f_vectors = [[new_individual.fitness.values[0], len(new_individual.pipeline)] for new_individual in
                         list_aux_save_individuals]
            x_vectors = []
            for i in list_aux_save_individuals:
                instance_individual_to_vectors = IndividuoVector()
                new_vector = instance_individual_to_vectors(i)
                x_vectors.append(new_vector)

            # Let's compute n_{i}
            n_i = math.ceil(number_of_configurations * (reduction_factor ** (
                -rung)))  # In the paper of Successive is ni, number of configurations to use in the next step
            n_i = int(4 * math.floor(n_i / 4))  # Because NSGA2 works with population multiples of 4
            if n_i < 8: n_i = 8  # Since we need a multiple of 4 for running the NSGA2 and at least 5 individuals, the closest number is 8
            # # From the number of configurations we need to look for a number multiple of 4 and can be divided into all the islands
            # elements_to_include_per_island = int(4 * round(n_i / 4))
            # if number_of_configurations < elements_to_include_per_island: elements_to_include_per_island-4
            # # In the next line we need to replace "n_i" for "len(list_aux_save_individuals)-1"
            number_subpopulations = math.floor(n_i / minimum_number_individuals_per_island) # We want to decrease the number of elements by rung
            # # "number_subpopulations" can be greather than "number_subpopulations" that is why
            # # we are going to use "minimum_number_of_islands"

            ops.evaluate = partial(ops.evaluate, subsample=rung_resources[rung])
            prob = pg.problem(AutoMLProblem(ops, path, rung, max_rung))
            population_list = [pg.population(prob) for i in range(minimum_number_of_islands)]
            isls = []
            replacement_policy_values = np.linspace(0.1, 0.9, num=minimum_number_of_islands)
            for subpop in range(minimum_number_of_islands): # 0 in range(16)
                # r_policy should be lower to greater
                r_policy = pg.r_policy(pg.fair_replace(rate=replacement_policy_values[subpop]))  # Share from 10% to 90% of the individulas en each island
                s_policy = pg.s_policy(udsp=pg.select_best())
                if number_subpopulations >= minimum_number_of_islands:
                    subpopulation_x = np.array(x_vectors)[subpop * minimum_number_individuals_per_island:(subpop + 1) * minimum_number_individuals_per_island]
                    subpopulation_f = np.array(f_vectors)[subpop * minimum_number_individuals_per_island:(subpop + 1) * minimum_number_individuals_per_island]
                elif number_subpopulations < minimum_number_of_islands:
                    # If the population is smaller than "minimum_number_individuals_per_island" we will generate use 8 random models
                    if number_subpopulations < minimum_number_individuals_per_island:
                        subpopulation_x = small_population_surrogate_x.copy()
                        subpopulation_f = small_population_surrogate_f.copy()
                    else: # if 3 < 4
                        if subpop<=number_subpopulations:
                            subpopulation_x = np.array(x_vectors)[subpop * minimum_number_individuals_per_island:(subpop + 1) * minimum_number_individuals_per_island]
                            subpopulation_f = np.array(f_vectors)[subpop * minimum_number_individuals_per_island:(subpop + 1) * minimum_number_individuals_per_island]
                        else:
                            subpopulation_x = small_population_surrogate_x.copy()
                            subpopulation_f = small_population_surrogate_f.copy()
                for i in range(len(subpopulation_x)):
                    if subpopulation_f[i][0] == -np.inf:
                        subpopulation_f[i][0] = -10000
                    population_list[subpop].push_back(x=subpopulation_x[i], f=[-subpopulation_f[i][0], subpopulation_f[i][1]])
                this_island = pg.island(algo=pg.algorithm(pg.nsga2(gen=iters)),
                                        pop=population_list[subpop],
                                        r_pol=r_policy,
                                        s_pol=s_policy,
                                        )
                isls.append(this_island)

            # Remove .pkl files (previous individuals)
            for root, dirs, files, in os.walk(path):
                for file in files:
                    if file.endswith(".pkl"):
                        name_file = path + "/" + file
                        os.remove(name_file)

            archi = pg.archipelago() # The topology it is not important here, it's only to initialize the archipelago correctly
            for isl in isls:
                archi.push_back(isl)
            # G = obtain_topology(name='complete_graph', nodes=int(len(archi)))
            # G = obtain_topology(name='circular_ladder_graph', nodes=int(len(archi) / 2))
            # G = obtain_topology(name = 'balanced_tree', nodes=int(len(archi)-1), h = 1)
            # G = obtain_topology(name = 'wheel', nodes=int(len(archi)))
            # G = obtain_topology(name='ladder_graph', nodes=int(len(archi) / 2))
            # G = obtain_topology(name = 'grid_graph', dim = (4, 4))
            # G = obtain_topology(name = 'grid_graph', dim = (4, 2, 2))
            # G = obtain_topology(name = 'hypercube_graph', nodes = 4)
            # G = obtain_topology(name = 'watts_strogatz_graph', nodes = int(len(archi)), k=2, p=0.5)
            # G = obtain_topology(name='cycle_graph', nodes=int(len(archi)))
            
            # # Small number of nodes (only 8 islands)
            G = obtain_topology(name = 'hypercube_graph', nodes = 3)
            # G = obtain_topology(name = 'grid_graph', dim = (4, 2))
            # G = obtain_topology(name = 'grid_graph', dim = (2, 2, 2))    
            
            this_topology = pg.free_form(G)
            # this_topology = pg.topology(pg.ring(len(archi)))
            archi.set_topology(this_topology)
            # print("Acabo de CREAR EL ARCHIPELAGO, EMPEZARÉ A EVOLUCIONAR EN PARALELO")

            # archi = pg.archipelago(n=islands, algo=algo, pop=pop, t=pg.topology(pg.ring()))
            # print("CREATION OF THE ARCHIPELAGO, IT WILL START THE EVOLUTION IN PARALLEL")
            # print(archi)
            #archi.get_champions_f()
            # print(archi.get_champions_f())
            archi.evolve()
            # archi.wait()
            archi.wait_check()
            #archi.get_champions_f()
            # print(archi.get_champions_f())
            # print("IT JUST FINISH")
            # print(archi)
            # print("Let's start with the iterative process")
            # print("len archi.get_champions_f()", len(archi.get_champions_f()))
            # print("len archi.get_champions_x()[0]", len(archi.get_champions_x()[0]))
            # print("len archi.get_champions_x()", len(archi.get_champions_x()))

            # final_output = []
            #x_of_island_champion = archi.get_champions_x()
            x_of_island_champion = []
            for isl in archi:
                this_population = isl.get_population()
                ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(this_population.get_f())
                best_ind = pg.select_best_N_mo(points=this_population.get_f(), N=10)
                x_of_island_champion += list(this_population.get_x()[best_ind])
            # print("El archipelago tiene ", len(x_of_island_champion), " nuevos individuos")
            for k in x_of_island_champion:
                final_instance = ValuesSearchSpace(k)
                individual_from_x = final_instance.get_individuals()
                result = ops.evaluate(individual_from_x)
                new_ind = result.individual
                if rung == max_rung:
                    output.append(new_ind)
            list_to_save_archipelagos.append(archi)
    # Let's save the different architectures
    path_save_archi = path + "/" + "archipelagos_during_process"
    with open(path_save_archi, 'wb') as f:
        pickle.dump(list_to_save_archipelagos, f)
    print("I finished the whole process, indeed this is cinema")
    return output


def evaluate_on_rung(individual, rung, max_rung, evaluate_individual, *args, **kwargs):
    # evaluation = evaluate_individual(individual, subsample=100, *args, **kwargs)
    evaluation = evaluate_individual(individual, *args, **kwargs)
    evaluation.individual.meta["rung"] = rung
    evaluation.individual.meta["subsample"] = kwargs.get("subsample")
    # We want to avoid saving evaluations that are not on the max rung to disk,
    # because we only want to use pipelines evaluated on the max rung after search.
    # We're working on a better way to relay this information, this is temporary.
    if evaluation.error is None and rung != max_rung:
        evaluation.error = "Not a full evaluation."
    return evaluation
