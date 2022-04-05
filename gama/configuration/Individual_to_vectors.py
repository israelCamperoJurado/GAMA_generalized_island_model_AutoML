import pygmo as pg
prob = pg.problem(pg.rosenbrock(dim = 4))
pop1 = pg.population(prob)
pop2 = pg.population(prob, size = 5, seed= 723782378)

pop1.push_back(x = [0.2,0.3,1.3,0.2], f = [15.2])
pop2.push_back(x = [0.2,0.3,1.3,0.2], f = [11.2])

pop1.set_xf(0, [1.,2.,3.,4.], [11.2]) # Ese indice ya debe de existir

#%%

import pygmo as pg
# The problem
prob = pg.problem(pg.rosenbrock(dim = 10))
# The initial population
pop = pg.population(prob, size=10)
best_fitness = pop.get_f()[pop.best_idx()]
print(best_fitness) 
for i in range(20):
    pop.push_back(x = [5.61530724,  0.85074663,  5.43011858,  7.73627579, -1.95986934,
        8.10397025,  5.76605759,  4.54661326, -3.76109341, -1.65934192], f = [1053669.2630184])
# pop.get_x()[0]
# pop.get_f()[0]
# The algorithm (a self-adaptive form of Differential Evolution (sade - jDE variant)
algo = pg.algorithm(pg.sade(gen = 1000))
# The actual optimization process
pop = algo.evolve(pop)
# Getting the best individual in the population
best_fitness = pop.get_f()[pop.best_idx()]
print(best_fitness) 
#%%

import pygmo as pg
# The problem
prob = pg.problem(pg.rosenbrock(dim = 10))
# The initial population
pop = pg.population(prob, 20)
for i in range(20):
    pop.push_back(x = [5.61530724,  0.85074663,  5.43011858,  7.73627579, -1.95986934,
        8.10397025,  5.76605759,  4.54661326, -3.76109341, -1.65934192], f = [1053669.2630184])
# pop.get_x()[0]
# pop.get_f()[0]
# The algorithm (a self-adaptive form of Differential Evolution (sade - jDE variant)
algo = pg.algorithm(pg.sade(gen = 1000))
# The actual optimization process
pop = algo.evolve(pop)
# Getting the best individual in the population
best_fitness = pop.get_f()[pop.best_idx()]
print(best_fitness) 

#%%
# Insert a population in a archipelago

import pygmo as pg

    
if __name__ == '__main__':
    algo = pg.algorithm(pg.sade(gen = 2000)) #genetic algorithm
    prob = pg.problem(pg.rosenbrock(dim = 10))
    pop = pg.population(prob, 30)
    # for i in range(20):
    #     pop.push_back(x = [5.61530724,  0.85074663,  5.43011858,  7.73627579, -1.95986934,
    #         8.10397025,  5.76605759,  4.54661326, -3.76109341, -1.65934192], f = [1053669.2630184])
    archi = pg.archipelago(n=32,algo=algo, pop=pop)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    #archi = pg.archipelago(n=32,algo=algo, prob=prob, pop=pop)
    #f_champ = archi.get_champions_f()
    #print(f_champ)
    # for i in f_champ
    #     print(i)
    # for i in range(1,200):
    #  archi.evolve(1);
    #  archi.wait()
    archi.evolve() 
    archi.wait()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
    
#%%

import pygmo as pg
    
if __name__ == '__main__':
    algo = pg.algorithm(pg.sade(gen = 2000)) #genetic algorithm
    prob = pg.problem(pg.rosenbrock(10))
    archi = pg.archipelago(t=pg.topology(pg.ring()))
    isls = [pg.island(algo=algo, prob=prob, size=20, udi=pg.thread_island()) for i in range(32)]
    for isl in isls:
        archi.push_back(isl)
    archi.evolve() 
    archi.wait()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
#%%

import pygmo as pg
    
if __name__ == '__main__':
    algo = pg.algorithm(pg.sade(gen = 2000)) #genetic algorithm
    prob = pg.problem(pg.rosenbrock(10))
    archi = pg.archipelago(t=pg.topology(pg.ring()))
    isls = [pg.island(algo=algo, prob=prob, size=20, udi=pg.thread_island()) for i in range(32)]
    for isl in isls:
        archi.push_back(isl)
    for i in range(500): 
        archi.evolve(1)
        archi.wait()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
#%%

import pygmo as pg
    
if __name__ == '__main__':
     #genetic algorithm
    prob = pg.problem(pg.rosenbrock(10))
    archi = pg.archipelago(t=pg.topology(pg.ring()))
    isl1 = pg.island(algo = pg.algorithm(pg.de(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = 2000)), prob=prob, size=20, udi=pg.thread_island())
    isls = [isl1, isl2, isl3, isl4, isl5, isl6, isl7, isl8]
    for isl in isls:
        archi.push_back(isl)
    for i in range(500): 
        archi.evolve(1)
        archi.wait()
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
    
#%%

import pygmo as pg

    
if __name__ == '__main__':
    algo = pg.algorithm(pg.sade(gen = 500)) #genetic algorithm
    prob = pg.problem(pg.rosenbrock(dim = 10))
    pop = pg.population(prob, 30)
    # for i in range(20):
    #     pop.push_back(x = [5.61530724,  0.85074663,  5.43011858,  7.73627579, -1.95986934,
    #         8.10397025,  5.76605759,  4.54661326, -3.76109341, -1.65934192], f = [1053669.2630184])
    archi = pg.archipelago(t=pg.topology(pg.ring()))
    isl1 = pg.island(algo = pg.algorithm(pg.de(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = 5000)), pop=pop, udi=pg.thread_island())
    isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = 5000)), pop=pop, udi=pg.thread_island())
    isls = [isl1, isl2, isl3, isl4, isl5, isl6, isl7, isl8]
    for isl in isls:
        archi.push_back(isl)
    print("Isla al inicio", archi)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    #archi = pg.archipelago(n=32,algo=algo, prob=prob, pop=pop)
    #f_champ = archi.get_champions_f()
    #print(f_champ)
    # for i in f_champ
    #     print(i)
    # for i in range(1,200):
    #  archi.evolve(1);
    #  archi.wait()
    archi.evolve() 
    archi.wait()
    print("Isla al final", archi)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
#%%


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
    algo = pg.algorithm(pg.sade(gen = 500)) #genetic algorithm
    prob = pg.problem(toy_problem(dim = 10))
    pop = pg.population(prob, 30)
    # for i in range(20):
    #     pop.push_back(x = [5.61530724,  0.85074663,  5.43011858,  7.73627579, -1.95986934,
    #         8.10397025,  5.76605759,  4.54661326, -3.76109341, -1.65934192], f = [1053669.2630184])
    archi = pg.archipelago(t=pg.topology(pg.ring()))
    isl1 = pg.island(algo = pg.algorithm(pg.de(gen = 5000)), pop=pop)
    isl2 = pg.island(algo = pg.algorithm(pg.sade(gen = 5000)), pop=pop)
    isl3 = pg.island(algo = pg.algorithm(pg.de1220(gen = 5000)), pop=pop)
    isl4 = pg.island(algo = pg.algorithm(pg.gwo(gen = 5000)), pop=pop)
    isl5 = pg.island(algo = pg.algorithm(pg.pso(gen = 5000)), pop=pop)
    isl6 = pg.island(algo = pg.algorithm(pg.pso_gen(gen = 5000)), pop=pop)
    isl7 = pg.island(algo = pg.algorithm(pg.sea(gen = 5000)), pop=pop)
    isl8 = pg.island(algo = pg.algorithm(pg.bee_colony(gen = 5000)), pop=pop)
    isls = [isl1, isl2, isl3, isl4, isl5, isl6, isl7, isl8]
    for isl in isls:
        archi.push_back(isl)
    print("Isla al inicio", archi)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    #archi = pg.archipelago(n=32,algo=algo, prob=prob, pop=pop)
    #f_champ = archi.get_champions_f()
    #print(f_champ)
    # for i in f_champ
    #     print(i)
    # for i in range(1,200):
    #  archi.evolve(1);
    #  archi.wait()
    archi.evolve() 
    archi.wait()
    print("Isla al final", archi)
    archi.get_champions_f() 
    print(archi.get_champions_f()) 
    
    
#%%

import pygmo as pg
from pygmo import *

# class raise_exception:
#     def __init__(self):
#         self.counter=0;
#     def fitness(self,dv):
#         #if self.counter == 300:
#         #    raise ValueError("Ops!")
#         self.counter += 1
#         f1 = dv[0]**2
#         return [f1 ]
#     def get_bounds(self):
#         return ([0],[1])
#     def get_name(self):
#         return "A throwing UDP"
class sphere_function:
    def fitness(self, x):
        return [sum(x*x)]

    def get_bounds(self):
        return ([-1,-1],[1,1])
    
isl = pg.island(algo = pg.de(100), prob = sphere_function(), size=20, udi=pg.ipyparallel_island())

print(isl.get_population().champion_f)

isl.evolve()
isl.wait()
isl.wait_check() 
print(isl.get_population().champion_f)

#%%

class toy_problem:
    def __init__(self, dim):
        self.dim = dim
        self.counter=0;

    def fitness(self, x):
        self.counter += 1
        #print(self.counter)
        if self.counter==50000:
            raise ValueError("Ops!")
        return [sum(x), 1 - sum(x*x), - sum(x)]

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x) # numerical gradient

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "A toy problem"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

import pygmo as pg
a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=1000))
p_toy = pg.problem(toy_problem(50))
p_toy.c_tol = [1e-4, 1e-4]
archi = pg.archipelago(n=10,algo=a_cstrs_sa, prob=p_toy, pop_size=70)
print(archi) 
print(archi.get_champions_f())
archi.evolve() 
archi.wait() 
print(archi) 
print(archi.get_champions_f())