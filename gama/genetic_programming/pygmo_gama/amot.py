# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 21:25:57 2021

@author: 20210595
"""

#Atomated Machine Learning Orientation for Time Series (AMOT)

import mo_amot
import ts_models

class AmotTimeSeries(object):
    """"Main class of AMOT TIME SERIES"""
    def __init__(self, search='MultiObjective', iterations=20, pop_size=20, islands=8):
        self.search = search
        self.iterations = iterations
        self.pop_size = pop_size
        self.islands = islands

    def fit(self, serie):
        if self.search == 'MultiObjective': #NSGA2
            x_best = mo_amot.main(serie=serie, 
                                  generations=4, 
                                  pop_size=40)
        instanceChoose = ts_models.ChoosePipeline()
        finalPipeline = instanceChoose(x=x_best, serie=serie)
        preprocessing = finalPipeline[0]
        timeSeriesModel = finalPipeline[1]
        model = timeSeriesModel[0].fit(disp=0)
        realTimSeries = preprocessing.fit(serie).transform(serie)
        predictTimeSeries = model.predict()
        print('Real Time Series:', realTimSeries)
        print('Predict Time Seires:', predictTimeSeries)
        return realTimSeries, predictTimeSeries, finalPipeline
    
        
if __name__=='__main__':    
    #from classifier_sklearn import x_va,nueva
    print("Y si hacemos un muneco")

#single_amot.israel("gato")



# def main(x_train, y_train, n_samples = 20, n_features=20, iters=1000, pop_size=70):
#     a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=iters))
#     prob = pg.problem(AutoMLProblem(x_train=x_train, y_train=y_train, features=n_features, instances=n_samples))
#     archi = pg.archipelago(n=32,algo=a_cstrs_sa, prob=prob, pop_size=pop_size)
#     archi.evolve() 
#     archi.wait()
#     a = archi.get_champions_f()
#     a2 = sorted(archi.get_champions_f(), key = lambda x: x[0])[0]
#     best_isl_idx = [(el == a2).all() for el in a].index(True)
#     x_best = archi.get_champions_x()[best_isl_idx]
#     f_best = archi.get_champions_f()[best_isl_idx]
#     print(x_best)
#     print(f_best)
