from classification import clf_config

upperBound = []
lowerBound = []

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
        # print(i, len(clf_config[i]))
        for h in clf_config[i].items():
            # print(h[0])
            if h[0] == 'alpha' and count_alpha == 2:
                upperBound.append(0.5)
                lowerBound.append(0) 
            if h[0] == 'alpha' and (count_alpha==0 or count_alpha==1):
                upperBound.append(100.0)
                lowerBound.append(1e-3)
                count_alpha += 1 
            if h[0] == 'fit_prior':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'min_samples_split':
                upperBound.append(21)
                lowerBound.append(2)
            if h[0] == 'min_samples_leaf':
                upperBound.append(21)
                lowerBound.append(1)
            if h[0] == 'max_depth':
                upperBound.append(11)
                lowerBound.append(1)
            if h[0] == 'criterion':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'max_features':
                upperBound.append(1)
                lowerBound.append(0.05)
            if h[0] == 'bootstrap':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'n_estimators':
                upperBound.append(200)
                lowerBound.append(100)
            if h[0] == 'learning_rate':
                upperBound.append(1.0)
                lowerBound.append(1e-3)
            if h[0] == 'subsample':
                upperBound.append(1)
                lowerBound.append(0.05)
            if h[0] == 'n_neighbors':
                upperBound.append(51)
                lowerBound.append(1)
            if h[0] == 'weights':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'p':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'penalty' and count_penalty==1:
                upperBound.append(0.4)
                lowerBound.append(0)
            if h[0] == 'penalty' and count_penalty==0:
                upperBound.append(1)
                lowerBound.append(0)
                count_penalty += 1
            if h[0] == 'loss':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'dual':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'tol':
                upperBound.append(1e-1)
                lowerBound.append(1e-5)
            if h[0] == 'C':
                upperBound.append(25.0)
                lowerBound.append(1e-4)
            if h[0] == 'solver':
                upperBound.append(0.4)
                lowerBound.append(0)
            if h[0] == 'threshold':
                upperBound.append(1)
                lowerBound.append(0.05)
            if h[0] == 'linkage':
                upperBound.append(2)
                lowerBound.append(0)
            if h[0] == 'affinity':
                upperBound.append(5)
                lowerBound.append(0)
            if h[0] == 'norm':
                upperBound.append(2)
                lowerBound.append(0)
            if h[0] == 'kernel':
                upperBound.append(8)
                lowerBound.append(0)
            if h[0] == 'gamma':
                upperBound.append(1)
                lowerBound.append(0)
            if h[0] == 'n_components':
                upperBound.append(11)
                lowerBound.append(1)
            if h[0] == 'svd_solver':
                upperBound.append(0.4)
                lowerBound.append(0)
            if h[0] == 'iterated_power':
                upperBound.append(11)
                lowerBound.append(1)
            if h[0] == 'degree':
                upperBound.append(2.4)
                lowerBound.append(2)
            if h[0] == 'include_bias':
                upperBound.append(0.4)
                lowerBound.append(0)
            if h[0] == 'interaction_only':
                upperBound.append(0.4)
                lowerBound.append(0)
            if h[0] == 'percentile':
                upperBound.append(100)
                lowerBound.append(1)
            # if h[0] == 'score_func':
            #     upperBound.append(0.4)
            #     lowerBound.append(0)
            # tonteria = len(upperBound)
            # print(lowerBound[tonteria-1], upperBound[tonteria-1])


import pygmo as pg
from pygmo import *

class AutoMLProblem:

    def __init__(self):
        self.op_pygmo = None

    # Define objectives
    def fitness(self, x):
        f1 = x[0]**2
        f2 = x[1]**2
        return [f1, f2]
    
    # Return number of objectives
    def get_nobj(self):
        return 2

    def get_bounds(self):
        lower = lowerBound
        upper = upperBound
        return (lower, upper)
    # Return function name
    def get_name(self):
        return "AutoMLProblem"
 
    
# prob = pg.problem(AutoMLProblem())
# algo = pg.algorithm(pg.cstrs_self_adaptive(iters=40))
# pop = pg.population(prob, 10)
# pop = algo.evolve(pop)
# print(pop.champion_f)

# create problem
prob = pg.problem(AutoMLProblem())
# create population
pop = pg.population(prob, size=20)
# select algorithm
algo = pg.algorithm(pg.nsga2(gen=40))
# run optimization
pop = algo.evolve(pop)
# extract results
fits, vectors = pop.get_f(), pop.get_x()
    
# def main(iters=10, pop_size=40):
#     a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=iters))
#     prob = pg.problem(AutoMLProblem())
#     archi = pg.archipelago(n=8,algo=a_cstrs_sa, prob=prob, pop_size=pop_size)
#     archi.evolve() 
#     archi.wait()
#     a = archi.get_champions_f()
#     a2 = sorted(archi.get_champions_f(), key = lambda x: x[0])[0]
#     best_isl_idx = [(el == a2).all() for el in a].index(True)
#     x_best = archi.get_champions_x()[best_isl_idx]
#     f_best = archi.get_champions_f()[best_isl_idx]
#     print(x_best)
#     print(f_best)
#     return x_best, f_best

# if __name__ == '__main__':
#     islands = 8
#     pop_size = 20
#     generations = 10
#     # select algorithm
#     algo = pg.algorithm(pg.nsga2(gen=generations))
#     # select problem
#     prob = pg.problem(AutoMLProblem())
#     # create an archipelago
#     archi = pg.archipelago(n=islands, algo=algo, prob=prob, pop_size=pop_size)
#     archi.evolve()
#     archi.wait()
#     listWithBestFitnessForIsland = [archi.get_migrants_db()[i][2][0][0] for i in range(islands)]
#     index_min = min(range(len(listWithBestFitnessForIsland)), key=listWithBestFitnessForIsland.__getitem__)
#     x_best = archi.get_migrants_db()[index_min][1][0]
#     f_best = archi.get_migrants_db()[index_min][2]
#     print(x_best)
#     print(f_best)
