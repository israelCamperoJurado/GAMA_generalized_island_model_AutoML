# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:47:26 2021

@author: 20210595
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:42:01 2021

@author: 20210595
"""


from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
from gama.search_methods.random_search import RandomSearch
from gama.search_methods.asha import AsynchronousSuccessiveHalving
# from gama.search_methods.rf9 import RandomForestTry
from gama.search_methods.pygmo_search import SearchPygmo # gama.py ya está modificado para trabajar perron sin que lo declaremos
#from gama.search_methods.pygmo_search4 import SearchPygmo #Funcionó mas o menos
from gama.search_methods.async_ea import AsyncEA
import pandas
import sys
import psutil # Search_pygmo

def on_terminate(proc):
    print("process {} terminated with exit code {}".format(proc, proc.returncode))
    
def kill_process_and_children(pid: int, sig: int = 15):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess as e:
        # Maybe log something here
        return

    for child_process in proc.children(recursive=True):
        child_process.send_signal(sig)
    proc.send_signal(sig)
    
if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    #data = pandas.read_csv("iris.csv")
    #data.head() # to see first 5 rows
    #X = data.drop(["species"], axis = 1)
    #y = data["species"]
    #print(X)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    #El tiempo que gama tiene por default son 6 minutos
    ####automl = GamaClassifier(max_total_time=15, store="nothing", search=SearchPygmo())
    #automl = GamaClassifier(max_total_time=60000, store="nothing", search=SearchPygmo(), n_jobs=-1) # n_jobs=-1
    automl = GamaClassifier(max_total_time=600, store="nothing", n_jobs=-1)
    #automl = GamaClassifier(max_total_time=120, store="nothing", n_jobs=-1) # Ya también es pygmo
    #automl = GamaClassifier(max_total_time=120, store='models') 
    #automl = GamaClassifier(max_total_time=30, store="nothing", search=RandomSearch())
    #automl = GamaClassifier(max_total_time=60, store="nothing", search=AsyncEA())
    # automl = GamaClassifier(max_total_time=10, store="nothing", search=RandomForestTry()) # Hemos usado rf9
    #automl = GamaClassifier(max_total_time=60, store="nothing", search=AsynchronousSuccessiveHalving())
    print("Starting `fit` which will take roughly 10 seconds.")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print('accuracy:', accuracy_score(y_test, label_predictions))
    print('log loss:', log_loss(y_test, probability_predictions))
    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
    print('log_loss', automl.score(X_test, y_test))
    
    
    
    # import psutil

    # kill_process_and_children(automl.pid)
    
    

    print("vamos a matar procesos")
    # procs = psutil.Process().children()
    # for p in procs:
    #     p.terminate()
        
    # gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)
    # for p in alive:
    #     p.kill()
    # print("Ya maté a todos los procesos")
    
    import psutil

    def on_terminate(proc):
        print("process {} terminated with exit code {}".format(proc, proc.returncode))
    
    procs = psutil.Process().children()
    for p in procs:
        p.terminate()
    print("terminé for p in procs")
    gone, alive = psutil.wait_procs(procs, timeout=1, callback=on_terminate)
    print("alive", alive)
    print("gone", gone)
    for p in alive:
        print("Entré al for p in alive")
        print(p)
        print(p)
        p.kill()
    print("pase el for p in alive")
        



