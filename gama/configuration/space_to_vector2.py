# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 19:17:30 2021

@author: 20210595
"""

from classification import clf_config
from gama.genetic_programming.components.individual import Individual
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components.primitive_node import PrimitiveNode
from gama.genetic_programming.components.primitive import Primitive
from gama.genetic_programming.components.terminal import Terminal

from sklearn.naive_bayes import MultinomialNB

upperBound = []
lowerBound = []
vector_support = []
# otro = []
# for i in clf_config:
#     if callable(i):
#         otro.append(100) # Choose the value that you want major than 50
#         print(i, len(clf_config[i]))
#         for k in range(len(clf_config[i])):
#             otro.append(1)
# print(otro, len(otro))

count_penalty = 0
count_alpha = 0

for i in clf_config:
    if callable(i):
        # print(i)
        upperBound.append(100) # Choose the value that you want major than 50
        lowerBound.append(0)
        vector_support.append(1)
        # print(i, len(clf_config[i]))
        for h in clf_config[i].items():
            # print(h[0])
            if h[0] == 'alpha' and count_alpha == 2:
                upperBound.append(0.5)
                lowerBound.append(0) 
                vector_support.append(0)
            if h[0] == 'alpha' and (count_alpha==0 or count_alpha==1):
                upperBound.append(100.0)
                lowerBound.append(1e-3)
                vector_support.append(0)
                count_alpha += 1 
            if h[0] == 'fit_prior':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'min_samples_split':
                upperBound.append(21)
                lowerBound.append(2)
                vector_support.append(0)
            if h[0] == 'min_samples_leaf':
                upperBound.append(21)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'max_depth':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'criterion':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'max_features':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'bootstrap':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'n_estimators':
                upperBound.append(200)
                lowerBound.append(100)
                vector_support.append(0)
            if h[0] == 'learning_rate':
                upperBound.append(1.0)
                lowerBound.append(1e-3)
                vector_support.append(0)
            if h[0] == 'subsample':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'n_neighbors':
                upperBound.append(51)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'weights':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'p':
                upperBound.append(2)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'penalty' and count_penalty==1:
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'penalty' and count_penalty==0:
                upperBound.append(1)
                lowerBound.append(0)
                count_penalty += 1
                vector_support.append(0)
            if h[0] == 'loss':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'dual':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'tol':
                upperBound.append(1e-1)
                lowerBound.append(1e-5)
                vector_support.append(0)
            if h[0] == 'C':
                upperBound.append(25.0)
                lowerBound.append(1e-4)
                vector_support.append(0)
            if h[0] == 'solver':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'threshold':
                upperBound.append(1)
                lowerBound.append(0.05)
                vector_support.append(0)
            if h[0] == 'linkage':
                upperBound.append(2)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'affinity':
                upperBound.append(5)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'norm':
                upperBound.append(2)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'kernel':
                upperBound.append(8)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'gamma':
                upperBound.append(1)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'n_components':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'svd_solver':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'iterated_power':
                upperBound.append(11)
                lowerBound.append(1)
                vector_support.append(0)
            if h[0] == 'degree':
                upperBound.append(2.4)
                lowerBound.append(2)
                vector_support.append(0)
            if h[0] == 'include_bias':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'interaction_only':
                upperBound.append(0.4)
                lowerBound.append(0)
                vector_support.append(0)
            if h[0] == 'percentile':
                upperBound.append(100)
                lowerBound.append(1)
                vector_support.append(0)
            # if h[0] == 'score_func':
            #     upperBound.append(0.4)
            #     lowerBound.append(0)
            #     vector_support.append(0)
            # tonteria = len(upperBound)
            # print(lowerBound[tonteria-1], upperBound[tonteria-1])
    
import pygmo as pg
from pygmo import *

class AutoMLProblem:

    # Define objectives
    def fitness(self, x):
        f1 = x[0]**2
        return [f1]
    
    # Define bounds
    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    
    # Return function name
    def get_name(self):
        return "AutoMLProblem"
 
# algo = pg.algorithm(pg.de(gen = 500))
# algo.set_verbosity(100)
# prob = pg.problem(AutoMLProblem())
# pop = pg.population(prob, 20)
# pop = algo.evolve(pop) 

def main(iters=40, pop_size=10):
    algo = pg.algorithm(pg.de(gen=iters))
    prob = pg.problem(AutoMLProblem())
    archi = pg.archipelago(n=32,algo=algo, prob=prob, pop_size=pop_size)
    archi.evolve() 
    archi.wait()
    fitness_of_island_champion = archi.get_champions_f()
    x_of_island_champion = archi.get_champions_x()
    print(fitness_of_island_champion)
    print(len(x_of_island_champion))

prob = pg.problem(AutoMLProblem())
algo = pg.algorithm(pg.de(gen=40))
pop = pg.population(prob, 10)
export = pop.get_x()
pop = algo.evolve(pop)
print(pop.champion_f)

positions = [i for i in range(len(vector_support)) if vector_support[i] == 1]
positionClassifier = [positions[i] for i in range(10)] # El díez es porque las primeras diez técnicas son de clasificacion
positionPreprocess = [positions[i] for i in range(10, len(positions))] 


    
    
primitiveClassification = Primitive(['MultinomialNB.alpha', # Primitive.input
                                      'MultinomialNB.fit_prior'],
                                    "prediction", # Primitive.output
                                    MultinomialNB # Primitive.identifier
                                    )

terminal1 = Terminal(10.0, 'alpha', 'alpha')
terminal2 = Terminal(False, 'fit_prior', 'fit_prior')

terminalClassification = [terminal1, terminal2]

primitiveNodeClassification = PrimitiveNode(primitiveClassification, "data", terminalClassification)
#print(primitiveNodeClassification)

#Ahora crearemos un Individual

ind = Individual(primitiveNodeClassification, compile_individual)
