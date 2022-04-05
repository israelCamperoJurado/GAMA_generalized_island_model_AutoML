# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 20:29:02 2021

@author: 20210595
"""
# import random
# class Ejemplo(object):
#     guardarIndividuos = []
#     def __init__(self):
#         self.name = "aver"
        
#     def __call__(self, a, b):
#         suma = a + b
#         Ejemplo.guardarIndividuos.append(suma)
#         return suma
    
# instancea = Ejemplo()
# for i in range(50):
    
#     a = random.randint(1,20)
#     b = random.randint(1,20)
#     resultado = instancea(a,b)
    
# print(instancea.guardarIndividuos)
    
# #%%
# from sklearn.feature_selection import (
#     SelectFwe,
#     SelectPercentile,
#     f_classif,
#     VarianceThreshold,
# )
# from sklearn.datasets import load_breast_cancer
# from sklearn.feature_selection import SelectFwe, chi2
# X, y = load_breast_cancer(return_X_y=True)
# X.shape

# X_new = SelectFwe(alpha=0.01, score_func=f_classif).fit_transform(X, y)
# X_new.shape



#%%
# from gama.configuration.bounds_pygmo import (
#     upperBound, 
#     lowerBound
#     ) 

# import pygmo as pg
# from pygmo import *

# class AutoMLProblem:
#     save_ind = []
#     save_whiout_evaluation = []
#     contador = 0
#     # Define objectives
#     def fitness(self, x):
#         AutoMLProblem.contador+=1
#         print("Veces que entré al fitness", AutoMLProblem.contador)
#         f1 = x[0]*2
#         return [f1]
    
#     # Define bounds
#     def get_bounds(self):
#         lower = lowerBound
#         upper = upperBound
#         return (lower, upper)
    
#     # Return function name
#     def get_name(self):
#         return "AutoMLProblem"

 

# algo = pg.algorithm(pg.de(gen = 50))
# prob = pg.problem(AutoMLProblem())
# pop = pg.population(prob, 50)
# pop = algo.evolve(pop) 
# final_pop = pop.get_x()
# for i in final_pop:
#     print(i)

# import pygmo as pg

# class toy_problem:
#     contador = 0
#     def __init__(self, dim):
#         self.dim = dim

#     def fitness(self, x):
#         toy_problem.contador += 1
#         print("Numero de veces que entré al fitness", toy_problem.contador)
#         return [sum(x), 1 - sum(x*x), - sum(x)]

#     def gradient(self, x):
#         return pg.estimate_gradient(lambda x: self.fitness(x), x) # numerical gradient

#     def get_nec(self):
#         return 1

#     def get_nic(self):
#         return 1

#     def get_bounds(self):
#         return ([-1] * self.dim, [1] * self.dim)

#     def get_name(self):
#         return "A toy problem"

#     def get_extra_info(self):
#         return "\tDimensions: " + str(self.dim)
    
# if __name__ == '__main__':
#     a_cstrs_sa = pg.algorithm(pg.de1220(gen=1000)) #genetic algorithm
#     p_toy = pg.problem(toy_problem(50))
#     p_toy.c_tol = [1e-4, 1e-4]
#     archi = pg.archipelago(n=32,algo=a_cstrs_sa, prob=p_toy, pop_size=70)
#     f_champ = archi.get_champions_f() 
#     print(f_champ)
#     # for i in f_champ:
#     #     print(i)
#     archi.evolve() 
#     archi.wait()
#     archi.get_champions_f() 
#     print(archi.get_champions_f()) 
    
import pygmo as pg

class toy_problem:
    contador = 0
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        toy_problem.contador += 1
        print("Numero de veces que entré al fitness", toy_problem.contador)
        return [sum(x)]

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "A toy problem"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)
    
if __name__ == '__main__':
    algo = pg.algorithm(pg.de(gen = 1000))
    prob = pg.problem(toy_problem(50))
    archi = pg.archipelago(n=32,algo=algo, prob=prob, pop_size=80)
    f_champ = archi.get_champions_f() 
    print("antes de evolve", f_champ)
    # for i in f_champ:
    #     print(i)
    archi.evolve() 
    archi.wait()
    archi.wait_check()
    print(archi)
    f_of_island_champion = archi.get_champions_f() 
    x_of_island_champion = archi.get_champions_x()
    print("despues de evolve", f_of_island_champion)
    
   
# from sklearn.svm import LinearSVC
# from sklearn.datasets import load_breast_cancer


# X, y = load_breast_cancer(return_X_y=True)
# clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False)

# clf.fit(X, y)