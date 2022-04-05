# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:11:49 2021

@author: 20210595
"""

import numpy as np

space = dict()
preprocessing= {}
models = {}

preprocessing.update({'SimpleImputer': 2,
                        #SimpleImputer: strategy -> string{'mean', 'median', 'most_frequent', 'constant'}
                        #SimpleImputer: fill_value ->'float', numerical(1, 2.3, etc.)) 
                        })
                        
models.update({'AutoReg': 2,
               #AutoReg: lags -> int
               #AutoReg: trend -> str{‘n’, ‘c’, ‘t’, ‘ct’}
               'ARIMA': 7,
               #ARIMA: order_p
               #ARIMA: order_d
               #ARIMA: order_q
               #ARIMA: trend ->  str{‘n’,’c’,’t’,’ct’}
               #ARIMA: enforce_stationarity -> bool
               #ARIMA: enforce_invertibility -> bool, optional
               #ARIMA: concentrate_scale ->bool, optional
               'SARIMAX': 6,
               #SARIMAX: order_p
               #SARIMAX: order_d
               #SARIMAX: order_q
               #SARIMAX: trend ->  str{‘n’,’c’,’t’,’ct’}
               #SARIMAX: enforce_stationarity -> bool
               #SARIMAX: enforce_invertibility -> bool, optional    
               'ExponentialSmoothing': 6,
               #ExponentialSmoothing: trend -> bool
               #ExponentialSmoothing: damped_trend -> bool, optional
               #ExponentialSmoothing: seasonal -> int, optional (7 for days)
               #ExponentialSmoothing: initialization_method -> str {‘estimated’, ‘heuristic’, ‘legacy-heuristic’, ‘known’}
               #ExponentialSmoothing: initial_level -> float, optional
               #ExponentialSmoothing: initial_trend -> float, optional
               'Holt': 5,
               #Holt: exponential -> bool, optional
               #Holt: damped_trend -> bool, optional
               #Holt: initialization_method -> str {‘estimated’, ‘heuristic’, ‘legacy-heuristic’, ‘known’}
               #Holt: initial_level -> float, optional
               #Holt: initial_trend -> float, optional
               'SimpleExpSmoothing': 2,
               #SimpleExpSmoothing: initialization_method -> str {‘estimated’, ‘heuristic’, ‘legacy-heuristic’, ‘known’}
               #SimpleExpSmoothing: initial_level -> float, optional
               'ExponentialSmoothing': 6,
               #ExponentialSmoothing: trend -> bool, optional
               #ExponentialSmoothing: damped_trend -> bool, optional 
               #ExponentialSmoothing: seasonal -> int, optional
               #ExponentialSmoothing: initialization_method -> str {‘estimated’, ‘heuristic’, ‘legacy-heuristic’, ‘known’}
               #ExponentialSmoothing: initial_level -> float, optional
               #ExponentialSmoothing: initial_trend -> float, optional
               'ETSModel': 8,
               #ETSModel: error -> str {'add', 'mul'}
               #ETSModel: trend -> str {'add', 'mul'}
               #ETSModel: damped_trend -> bool, optional
               #ETSModel: seasonal -> str {'add', 'mul'}
               #ETSModel: seasonal_periods -> int, optional (7 for a week)
               #ETSModel: initialization_method -> str {‘estimated’, ‘heuristic’, ‘known’}
               #ETSModel: initial_level -> float, optional
               #ETSModel: initial_trend -> float, optional
               'DynamicFactor': 3,
               #DynamicFactor: k_factors -> int
               #DynamicFactor: factor_order -> int
               #DynamicFactor: error_cov_type -> str{‘scalar’, ‘diagonal’, ‘unstructured’}
               'DynamicFactorMQ': 7,
               #DynamicFactorMQ: k_endog_monthly -> int
               #DynamicFactorMQ: factors -> int
               #DynamicFactorMQ: factor_orders -> int 
               #DynamicFactorMQ: factor_multiplicities -> int
               #DynamicFactorMQ: idiosyncratic_ar1 -> bool
               #DynamicFactorMQ: standardize -> bool 
               #DynamicFactorMQ: init_t0 -> bool, optional
               'VAR': 0,
               'VARMAX': 7,
               #VARMAX: order_p -> int
               #VARMAX: order_q -> int
               #VARMAX: trend -> str{‘n’,’c’,’t’,’ct’}
               #VARMAX: error_cov_type -> str{‘diagonal’, ‘unstructured’}
               #VARMAX: measurement_error -> bool, optional
               #VARMAX: enforce_stationarity -> bool, optional
               #VARMAX: enforce_invertibility -> bool, optional
               'SVAR': 1,
               #SVAR: svar_typestr -> str{'A', 'B', 'AB'}
               'VECM': 3,
               #VECM: k_ar_diff -> int
               #VECM: coint_rank -> int
               #VECM: deterministic -> str {"nc", "co", "ci", "lo", "li"}
               'UnobservedComponents': 12
               #UnobservedComponents: level -> bool
               #UnobservedComponents: trend -> bool
               #UnobservedComponents: seasonal -> int
               #UnobservedComponents: cycle -> bool
               #UnobservedComponents: autoregressive -> int
               #UnobservedComponents: irregular -> bool
               #UnobservedComponents: stochastic_level -> bool
               #UnobservedComponents: stochastic_trend -> bool
               #UnobservedComponents: stochastic_seasonal -> bool
               #UnobservedComponents: stochastic_cycle > bool
               #UnobservedComponents: damped_cycle > bool
               #UnobservedComponents: use_exact_diffuse > bool
    })

space.update({'preprocessing': preprocessing, 
              'models': models})

#Define the dimension of the vector (dimensionVector) to the PYGMO problem

#print(space)

primitives1 = len([primitive for primitive in preprocessing.keys()])
primitives2 = len([primitive for primitive in models.keys()])
terminals1 = np.sum([terminals for terminals in preprocessing.values()])
terminals2 = np.sum([terminals for terminals in models.values()])
dimensionVector = np.sum([primitives1, primitives2, terminals1, terminals2])


#Assign positions to the vector for each primitive and terminal (to activate and deactive range of cells)
positions = list()
lower = 0
upper = 0
for i in preprocessing.keys():
    upper = lower + preprocessing[i] #Plus one for the primitivie itself
    positionPre = [i, lower, upper]
    positions.append(positionPre)
    lower = upper + 1
#Remember positions has 3 elements for each inner list, the second column is the position
#that represent the primitive and the third one is the range for its hyperparameters

for i in models.keys():
    upper = upper + models[i] + 1 #Plus one for the primitivie itself
    positionPre = [i, lower, upper]
    positions.append(positionPre)
    lower = upper + 1

positionsPreprocessingTechniques = [positions[i][1] for i in range(primitives1)] 

positionsModels = [positions[i][1] for i in range(primitives1, primitives1+primitives2)]
                        
float_int_positions =[['SimpleImputer', 'float'],
                      ['SimpleImputer_strategy', 'int'],
                      ['SimpleImputer_fill_value', 'float'],
                      ['AutoReg', 'float'],
                      ['AutoReg_lags', 'int'],
                      ['AutoReg_trend', 'int'],
                      ['ARIMA', 'float'],
                      ['ARIMA_order_p', 'int'],
                      ['ARIMA_order_d', 'int'],
                      ['ARIMA_order_q', 'int'],
                      ['ARIMA_trend', 'int'],
                      ['ARIMA_enforce_stationarity', 'int'],
                      ['ARIMA_enforce_invertibility', 'int'],
                      ['ARIMA_concentrate_scale', 'int'],
                      ['SARIMAX', 'float'],
                      ['SARIMAX_order_p', 'int'],
                      ['SARIMAX_order_d', 'int'],
                      ['SARIMAX_order_q', 'int'],
                      ['SARIMAX_trend', 'int'],
                      ['SARIMAX_enforce_stationarity', 'int'],
                      ['SARIMAX_enforce_invertibility', 'int'],
                      ['ExponentialSmoothing', 'float'],
                      ['ExponentialSmoothing_trend', 'int'],
                      ['ExponentialSmoothing_damped_trend', 'int'],
                      ['ExponentialSmoothing_seasonal', 'int'],
                      ['ExponentialSmoothing_initialization_method', 'int'],
                      ['ExponentialSmoothing_initial_level', 'float'],
                      ['ExponentialSmoothing_initial_trend', 'float'],
                      ['Holt', 'float'],
                      ['Holt_exponential', 'int'],
                      ['Holt_damped_trend', 'int'],
                      ['Holt_initialization_method', 'int'],
                      ['Holt_initial_level', 'float'],
                      ['Holt_initial_trend', 'float'],
                      ['SimpleExpSmoothing', 'float'],
                      ['SimpleExpSmoothing_initialization_method', 'int'],
                      ['SimpleExpSmoothing_initial_level', 'float'],
                      ['ETSModel', 'float'],
                      ['ETSModel_error', 'int'],
                      ['ETSModel_trend', 'int'],
                      ['ETSModel_damped_trend', 'int'],
                      ['ETSModel_seasonal', 'int'],
                      ['ETSModel_seasonal_periods', 'int'],
                      ['ETSModel_initialization_method', 'int'],
                      ['ETSModel_initial_level', 'float'],
                      ['ETSModel_initial_trend', 'float'],
                      ['DynamicFactor', 'float'],
                      ['DynamicFactor_k_factors', 'int'],
                      ['DynamicFactor_factor_order', 'int'],
                      ['DynamicFactor_error_cov_type', 'int'],
                      ['DynamicFactorMQ', 'float'],
                      ['DynamicFactorMQ_k_endog_monthly', 'int'],
                      ['DynamicFactorMQ_factors', 'int'],
                      ['DynamicFactorMQ_factor_orders', 'int'],
                      ['DynamicFactorMQ_factor_multiplicities', 'int'],
                      ['DynamicFactorMQ_idiosyncratic_ar1', 'int'],
                      ['DynamicFactorMQ_standardize', 'int'],
                      ['DynamicFactorMQ_init_t0', 'int'],
                      ['VAR', 'float'],
                      ['VARMAX', 'float'],
                      ['VARMAX_order_p', 'int'],
                      ['VARMAX_order_q', 'int'],
                      ['VARMAX_trend', 'int'],
                      ['VARMAX_error_cov_type', 'int'],
                      ['VARMAX_measurement_error', 'int'],
                      ['VARMAX_enforce_stationarity', 'int'],
                      ['VARMAX_enforce_invertibility', 'int'],
                      ['SVAR', 'float'],
                      ['SVAR_svar_typestr', 'int'],
                      ['VECM', 'float'],
                      ['VECM_k_ar_diff', 'int'],
                      ['VECM_coint_rank', 'int'],
                      ['VECM_deterministic', 'int'],
                      ['UnobservedComponents', 'float'],
                      ['UnobservedComponents_level', 'int'],
                      ['UnobservedComponents_trend', 'int'],
                      ['UnobservedComponents_seasonal', 'int'],
                      ['UnobservedComponents_cycle', 'int'],
                      ['UnobservedComponents_autoregressive', 'int'],
                      ['UnobservedComponents_irregular', 'int'],
                      ['UnobservedComponents_stochastic_level', 'int'],
                      ['UnobservedComponents_stochastic_trend', 'int'],
                      ['UnobservedComponents_stochastic_seasonal', 'int'],
                      ['UnobservedComponents_stochastic_cycle', 'int'],
                      ['UnobservedComponents_damped_cycle', 'int'],
                      ['UnobservedComponents_use_exact_diffuse', 'int']
                      ]