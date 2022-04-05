# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:33:05 2021

@author: 20210595
"""
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


class ScoreAmot(object):
    _count = 0
    def __init__(self, pipeline, serie):
        self.pipeline = pipeline 
        self.serie = serie
        ScoreAmot._count += 1
        self.id = ScoreAmot._count
        
    def evaluation(self):
        preprocessing = self.pipeline[0]
        timeSeriesModel = self.pipeline[1]
        print("Evaluating TimeSeries Pipeline in", self.id)
        if timeSeriesModel != 'invalid':
            try:
                print("Time Series, assessing a vector for the creation of a pipeline", self.id)
                model = timeSeriesModel[0].fit(disp=0)
                realTimSeries = preprocessing.fit(self.serie).transform(self.serie)
                predictTimeSeries = model.predict()
                mse = mean_squared_error(realTimSeries, predictTimeSeries, squared=False)
                #aic = model.aic
                print('Time Series pipeline valid, mse:', mse)
            except:
                mse = 1000000
        else:
            mse = 1000000
        return mse

if __name__=='__main__':  
    from ts_models import nueva
    print('I am in amot_validation')
    vector = []
    for i in nueva:
        nuevaIstancia = ScoreAmot(i)
        vector.append(nuevaIstancia.evaluation())
    print(vector)