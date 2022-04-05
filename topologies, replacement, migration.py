# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:49:52 2021

@author: 20210595
"""

class toy_problem:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
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
a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=10))
p_toy = pg.problem(toy_problem(50))
p_toy.c_tol = [1e-4, 1e-4]
r_policy = pg.r_policy(pg.fair_replace(rate=0.5))
s_policy = pg.s_policy(udsp=pg.select_best())
archi = pg.archipelago(n=32,algo=a_cstrs_sa, prob=p_toy, r_pol=r_policy, s_pol = s_policy,pop_size=70)
p2p = pg.migration_type(0)
broadcast = pg.migration_type(1)
archi.set_migration_type(broadcast)
t=pg.topology(pg.free_form())
print(archi) 

archi.evolve() 
print(archi) 



archi.wait()
archi.get_champions_f() 